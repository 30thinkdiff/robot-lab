"""Optional OGG objectives. Frozen cosine code is delegated to unchanged.

Raw MSE is intentional: no feature rescaling or eta tuning is hidden here.
Only the first clean candidate action is differentiable in this pilot.
"""
import torch
import torch.nn.functional as F

LAYERS = ('block_02', 'block_08', 'connector')
MODES = ('cosine', 'mse', 'mse_cosine')


def components(predicted, target, mode):
    if mode not in MODES:
        raise ValueError('unknown OGG loss')
    p, t = predicted.float().flatten(1), target.detach().float().flatten(1)
    if p.shape != t.shape or not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('invalid prediction/target')
    mse = (p - t).square().mean()
    cosine = (1 - F.cosine_similarity(p, t, dim=-1)).mean()
    loss = cosine if mode == 'cosine' else mse if mode == 'mse' else mse + cosine
    return loss, {'raw_mse': float(mse.detach()), 'cosine_loss': float(cosine.detach())}


def mlp_factory(original, mode, report=lambda row: None):
    if mode not in MODES:
        raise ValueError('unknown OGG loss')
    def guided(predictor, z_t, x_tau, velocity, timestep, target_delta, eta=1.):
        if mode == 'cosine':
            return original(predictor, z_t, x_tau, velocity, timestep, target_delta, eta=eta)
        if predictor.training or any(p.requires_grad for p in predictor.parameters()):
            raise ValueError('frozen eval predictor required')
        with torch.inference_mode(False), torch.enable_grad():
            v = velocity.detach().clone().requires_grad_(True)
            action = (x_tau.detach().clone() - timestep[:, None, None] * v)[:, 0, :predictor.cfg.action_dim]
            delta = predictor(z_t.detach().clone(), action)
            loss, terms = components(delta, target_delta, mode)
            gradient = torch.autograd.grad(loss, v)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('nonfinite OGG gradient')
            out = (v - eta * gradient).detach()
            report({'loss_mode': mode, 'loss': float(loss.detach()), **terms,
                    'gradient_l2': float(gradient.norm()), 'velocity_update_l2': float((out - velocity).norm())})
            return out, float(loss.detach())
    return guided


def joint_factory(original, mode, report=lambda row: None):
    if mode not in MODES:
        raise ValueError('unknown OGG loss')
    def guided(joint, context, x_t, velocity, timestep, targets, calibration, eta=1.):
        if mode == 'cosine':
            return original(joint, context, x_t, velocity, timestep, targets, calibration, eta=eta)
        if joint.training or any(p.requires_grad for p in joint.parameters()):
            raise ValueError('frozen eval J required')
        if velocity.shape != x_t.shape or velocity.ndim != 3 or velocity.shape[-1] < 7:
            raise ValueError('flow action shape')
        if timestep.shape != (velocity.shape[0],) or not torch.isfinite(timestep).all() or (timestep < 0).any() or (timestep > 1).any():
            raise ValueError('flow timestep')
        if not 0 < eta < 1e6:
            raise ValueError('invalid eta')
        with torch.inference_mode(False), torch.enable_grad():
            v = velocity.detach().clone().requires_grad_(True)
            action = (x_t.detach().clone() - timestep[:, None, None] * v)[:, 0, :7]
            out = joint(dict({k: value.detach().clone() for k, value in context.items()}, action=action))
            losses, terms = {}, {}
            for k in LAYERS:
                target = targets[k].detach().float().flatten(1)
                pred = out['delta_' + k].float().flatten(1)
                floor = max(calibration['norm_floor'][k], 1e-8)
                if (torch.isfinite(pred).all() and torch.isfinite(target).all()
                    and float(pred.detach().square().mean().sqrt()) >= floor
                    and float(target.square().mean().sqrt()) >= floor and calibration['weights'][k] > 0):
                    losses[k], terms[k] = components(pred, target, mode)
            if not losses:
                return velocity.detach(), {'skipped': 'no_reliable_correction_layer', 'gradient_norm': 0., 'updated': False}
            denom = sum(calibration['weights'][k] for k in losses)
            total = sum(calibration['weights'][k] / denom * loss for k, loss in losses.items())
            gradients = {k: torch.autograd.grad(loss, v, retain_graph=True)[0] for k, loss in losses.items()}
            gradient = torch.autograd.grad(total, v)[0]
            if not torch.isfinite(total) or not torch.isfinite(gradient).all():
                raise FloatingPointError('OGG loss/gradient')
            result = (v - eta * gradient).detach()
            details = {'loss_mode': mode, 'loss': float(total.detach()), 'per_layer_components': terms,
                'per_layer_loss': {k: float(loss.detach()) for k, loss in losses.items()},
                'per_layer_gradient_norm': {k: float(g.norm()) for k, g in gradients.items()},
                'gradient_cosines': {a+'/'+b: float(F.cosine_similarity(gradients[a].flatten(), gradients[b].flatten(), dim=0)) for i,a in enumerate(gradients) for b in list(gradients)[i+1:]},
                'gradient_norm': float(gradient.norm()), 'velocity_update_norm': float((result - velocity).norm()),
                'candidate_action_update_norm': float((timestep[:, None, None] * (result - velocity))[:, :, :7].norm()),
                'updated': bool(gradient.norm() > 0), 'valid_layers': list(losses)}
            report(details)
            return result, details
    return guided
