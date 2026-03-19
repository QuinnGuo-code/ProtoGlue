"""Attention-based fusion modules for ProtoGlue."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BaseAttention
from protoglue.config import TEMP_START


class TwoTokenGatedMHA(BaseAttention):
    """Intra-modality fusion: (spatial_emb, feature_emb) -> gated fused output.

    Uses multi-head attention over two tokens followed by a learned gate
    to produce a single fused representation per spot.
    """

    def __init__(self, d, num_heads=4, dropout=0.1, use_residual=True, use_norm=True):
        super().__init__(d, use_residual=use_residual, use_norm=use_norm)
        self.mha = nn.MultiheadAttention(
            d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.init_weights()

    def forward(self, a, b):
        H = torch.stack([self.norm(a), self.norm(b)], dim=1)
        H2, _ = self.mha(H, H, H, need_weights=False)
        H = H + H2
        H = H + self.ffn(H)
        a2, b2 = H[:, 0, :], H[:, 1, :]
        g = self.gate(torch.cat([a2, b2], dim=1))
        fused = g * a2 + (1.0 - g) * b2
        if self.use_residual:
            fused = fused + a
        alpha = torch.stack([g.mean(dim=1), 1.0 - g.mean(dim=1)], dim=1)
        return fused, alpha


class CrossAttentionGatedFusion(BaseAttention):
    """Inter-modality fusion with temperature-annealed gating.

    Performs bidirectional cross-attention between two modality embeddings,
    then combines them via a learned, temperature-controlled gate.
    """

    def __init__(self, d, num_heads=4, dropout=0.1, use_residual=True, use_norm=True):
        super().__init__(d, use_residual=use_residual, use_norm=use_norm)
        self.mha_ab = nn.MultiheadAttention(
            d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mha_ba = nn.MultiheadAttention(
            d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate_logit = nn.Linear(2 * d, 1)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.temp = float(TEMP_START)
        self.init_weights()

    def set_temp(self, t):
        """Set temperature for gating sharpness."""
        self.temp = float(max(1e-3, t))

    def forward(self, a, b):
        a_n, b_n = self.norm(a).unsqueeze(1), self.norm(b).unsqueeze(1)
        a2, _ = self.mha_ab(a_n, b_n, b_n, need_weights=False)
        b2, _ = self.mha_ba(b_n, a_n, a_n, need_weights=False)
        a2, b2 = a2.squeeze(1), b2.squeeze(1)
        if self.use_residual:
            a2, b2 = a + a2, b + b2
        a2, b2 = a2 + self.ffn(a2), b2 + self.ffn(b2)
        g = torch.sigmoid(
            self.gate_logit(torch.cat([a2, b2], dim=1)) / self.temp
        )
        fused = g * a2 + (1.0 - g) * b2
        return fused, torch.cat([g, 1.0 - g], dim=1)
