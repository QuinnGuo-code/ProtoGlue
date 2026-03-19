"""Basic layers used across the ProtoGlue architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    """Base class for attention modules with optional residual and layer norm."""

    def __init__(self, d, use_residual=True, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(d) if use_norm else nn.Identity()
        self.use_residual = use_residual

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ModalityProjector(nn.Module):
    """Projects modality embeddings to a shared fusion space."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = (
            nn.Identity()
            if int(in_dim) == int(out_dim)
            else nn.Sequential(
                nn.Linear(int(in_dim), int(out_dim)),
                nn.LayerNorm(int(out_dim)),
                nn.Dropout(dropout),
            )
        )

    def forward(self, x):
        return self.net(x)


class SparseGCN(nn.Module):
    """Single-layer graph convolution on sparse adjacency."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, A):
        h = self.lin(torch.sparse.mm(A, self.drop(x)))
        return self.norm(F.elu(h))
