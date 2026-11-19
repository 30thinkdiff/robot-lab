"""Metadata for existing Replay-v1 readers; does not create or import fake trials."""
from .baseline import SHAPES


def case_config(layer, mode, checkpoint_sha256):
    if layer not in SHAPES or mode not in ('monitor', 'lvm', 'ogg'):
        raise ValueError('Unknown single-layer method')
    if len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256):
        raise ValueError('Actual checkpoint SHA256 required')
    label = 'S2' if layer == 'block_02' else 'S8'
    return dict(model_label=label + ' / 原Residual MLP', feature_layer=layer,
        model_type='mlp', steps=50, mode=mode, uses_corrector=True,
        corrector_sha256=checkpoint_sha256, prediction_seconds=1.,
        data_fps=10., control_fps=20., first_score_step=24,
        ogg_loss_mode='cosine' if mode == 'ogg' else None,
        historical_data_read_only=True)


def recording_requirements():
    return dict(base_format='Replay-v1', motion_format='robotlab_motion_v1',
        required=['plan.json', 'execution.json', 'trials/<id>.json', 'actual_protocol.json',
                  'monitor_steps.jsonl', 'controller_commands.json', 'action_chunks',
                  'video', 'frame_map.jsonl', 'motion/protocol.json', 'motion/steps.jsonl',
                  'motion/status.json', 'motion/chunks'],
        video_phase='before_action', terminal_motion_state_required=True,
        missing_data='explicitly unrecorded; do not fabricate or infer physical measurements',
        motion_capture='reuse existing MotionCapture, validate capture on/off on actual GPU before deployment',
        real_gpu_validation_completed=False, replay_import_completed=False)
