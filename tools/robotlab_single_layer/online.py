"""Only replace visual observation sites; inherit original single-action controller."""
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .baseline import CAMERAS, SHAPES, digest, model_config
from dataclasses import asdict


def vision_tap_class(path):
    spec = importlib.util.spec_from_file_location('_single_layer_readonly_vision', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReadOnlyVisionTap


def load_predictor(reference, checkpoint, source_root, device):
    predictor, protocol = reference['runtime'].load_predictor(Path(checkpoint), Path(source_root), device)
    layers = {r.get('feature_layer') for r in protocol.get('records', [])}
    if len(layers) != 1 or not layers <= set(SHAPES):
        raise ValueError('Checkpoint lacks an unambiguous S2/S8 training provenance')
    expected = json.loads(json.dumps(asdict(model_config(reference))))
    if protocol['model'] != expected or protocol.get('model_type') != 'mlp' or protocol.get('precision') != 'amp-bfloat16':
        raise ValueError('Checkpoint architecture/precision differs from the historical MLP adaptation')
    return predictor, dict(protocol, visual_layer=layers.pop(),
                           checkpoint_sha256=digest(checkpoint))


def corrector_class(reference, tap_path):
    legacy = reference['runtime']
    Tap = vision_tap_class(tap_path)

    class SingleLayerCorrector(legacy.OnlineCorrector):
        def __init__(self, policy, predictor, protocol, method, source_root, directory, *, layer,
                     diagnostics=True, **kwargs):
            if layer not in SHAPES or predictor.cfg.token_dim != 768 or protocol.get('visual_layer') != layer:
                raise ValueError('S2/S8 require unchanged MLP with token_dim768')
            if predictor.training or any(p.requires_grad for p in predictor.parameters()):
                raise ValueError('Predictor must be eval and frozen')
            if getattr(policy.config, 'n_action_steps', None) != 50 or getattr(policy.config, 'num_steps', None) != 10:
                raise ValueError('Frozen policy must retain 50-step chunks and 10 denoising steps')
            required = dict(h_window=1, anchor_history=3, k_step=10, fps=10., prediction_seconds=1.)
            if any(protocol.get(k) != v for k, v in required.items()):
                raise ValueError('Baseline timebase/history protocol mismatch')
            for key, value in dict(fps=20, stride=1, eta=1., guidance_every=3,
                                   implementation='reference', cooldown=0, force_step=-1).items():
                if key in kwargs and kwargs[key] != value:
                    raise ValueError('Unapproved control change: ' + key)
                kwargs[key] = value
            self.layer, self.diagnostics = layer, diagnostics
            self.observed_prediction_origins = {}
            super().__init__(policy, predictor, protocol, method, source_root, directory, **kwargs)

        @torch.no_grad()
        def encode(self, batch):
            if list(self.policy.config.image_features) != CAMERAS:
                raise ValueError('Camera order mismatch')
            images, masks = self.policy.prepare_images(batch)
            if len(images) != 2 or any(not bool(m.all()) for m in masks):
                raise ValueError('Two complete views required')
            wrapper = self.policy.model.vlm_with_expert
            # The frozen loader keeps vision/Connector FP32 but some language
            # and action-expert tensors are BF16. They are not the visual tap.
            # Validate the actual visual path; never cast the policy to pass.
            vlm=wrapper.get_vlm_model()
            for name,module in (('vision',vlm.vision_model),('connector',vlm.connector)):
                dtypes={p.dtype for p in module.parameters() if p.is_floating_point()}
                if dtypes and dtypes!={torch.float32}:
                    raise ValueError('Historical online '+name+' FP32 required; no implicit dtype conversion')
            values = []
            with Tap(wrapper, (int(self.layer[-2:]),)) as tap:
                for camera, image in zip(CAMERAS, images, strict=True):
                    with tap.capture(camera) as captured:
                        wrapper.embed_image(image)  # same FP32 policy encoder; no additional VLA call
                    value = captured[self.layer]
                    if value.shape != (1, 1024, 768):
                        raise ValueError('Unpooled raw feature shape mismatch')
                    values.append(value.float())
            visual = torch.stack(values, dim=1)
            if visual.shape != (1, 2, 1024, 768) or not torch.isfinite(visual).all():
                raise FloatingPointError('Raw visual shape/numerics')
            return visual.reshape(1, 2048, 768).detach()

        def log(self, row):
            if not self.diagnostics:
                return legacy.OnlineCorrector.log(self, row)
            row = dict(row, feature_layer=self.layer)
            if 'score' in row:
                s = self.state
                origin = self.observed_prediction_origins.pop(self.step, None) if row['score'] is not None else None
                row.update(ewma=s.ewma_prev,
                    threshold_on=s.dynamic_threshold[-1] if s.dynamic_threshold else None,
                    threshold_off=s.dynamic_release_threshold[-1] if s.dynamic_release_threshold else None,
                    bootstrapped=s.bootstrapped, over_count=s.over_count, under_count=s.under_count,
                    hard_over_count=s.hard_over_count, cooldown_remaining=s.cooldown_remaining,
                    score_valid=row['score'] is not None, prediction_origin_step=origin,
                    prediction_due_step=self.step if row['score'] is not None else None,
                    pending_predictions=len(self.pending), action_queue_length=len(self.policy._queues['action']),
                    events_total=self.events, observation_phase='before_action',
                    queue_snapshot_phase='after_action_selection',
                    state_phase='after_threshold_update' if row['score'] is not None else 'no_threshold_update',
                    pending_prediction_due_steps=[item[0] for item in self.pending])
                if self.pending and self.pending[-1][0] == self.step + self.k:
                    due = self.pending[-1][0]
                    self.observed_prediction_origins[due] = self.step
                    row['prediction_created'] = dict(origin_step=self.step, due_step=due,
                        prediction_id=self.layer + ':' + str(self.step))
            legacy.OnlineCorrector.log(self, row)

        def denoise_step(self, *args, **kwargs):
            if not self.diagnostics:
                return super().denoise_step(*args, **kwargs)
            original = legacy.guided_velocity

            def observed_guidance(predictor, visual, x_tau, velocity, timestep, target, eta=1.):
                gradients = []
                def capture_action(module, inputs):
                    action = inputs[1]
                    if action.requires_grad:
                        action.register_hook(lambda g: gradients.append(g.detach().clone()))
                handle = predictor.register_forward_pre_hook(capture_action)
                started = time.perf_counter()
                try:
                    guided, loss = original(predictor, visual, x_tau, velocity, timestep, target, eta)
                finally:
                    handle.remove()
                self.log(dict(step=self.step, event='single_layer_ogg_details', denoise=self.denoise_index,
                    loss=loss, eta=eta, guidance_every=self.every,
                    action_gradient_l2=float(gradients[0].norm()) if gradients else None,
                    gradient_definition='actual d(loss)/d(candidate_first_action), not a parameter gradient',
                    velocity_update_l2=float((guided - velocity).norm()),
                    clean_action_update_l2=float((timestep[:, None, None] * (guided - velocity))[:, :, :7].norm()),
                    target_shape=list(target.shape), target_rms=float(target.square().mean().sqrt()),
                    latency_scope='guidance plus diagnostic copies/norms; scalar reads synchronize; excludes JSON write',
                    latency_s=time.perf_counter() - started))
                return guided, loss
            # Observe the same function call; do not recompute a model forward,
            # replace the loss, change eta, or return a modified gradient.
            with patch.object(legacy, 'guided_velocity', observed_guidance):
                return super().denoise_step(*args, **kwargs)

    return SingleLayerCorrector
