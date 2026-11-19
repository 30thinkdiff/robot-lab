"""Shape-only adapter over the exact historical Residual MLP; no new layers."""
import torch
from torch import nn


class LayerShapeAdapter(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.cfg = predictor.cfg
        if self.cfg.token_dim != 768 or self.cfg.action_dim != 7:
            raise ValueError('Single-layer protocol requires token_dim768, action_dim7')

    def forward(self, visual, action):
        if visual.ndim != 4 or tuple(visual.shape[1:]) != (2, 1024, 768):
            raise ValueError('Requires unpooled [B,2,1024,768]')
        if action.shape != (visual.shape[0], 7):
            raise ValueError('Requires one current action [B,7]')
        shape = visual.shape
        return self.predictor(visual.reshape(shape[0], 2048, 768), action).reshape(shape)
