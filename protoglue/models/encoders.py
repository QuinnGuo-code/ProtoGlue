"""Graph convolutional encoders for ProtoGlue."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SparseGCN


class ResidualGCN(nn.Module):
    """Multi-layer GCN with residual connections."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [
                SparseGCN(dims[i], dims[i + 1], dropout=dropout if i > 0 else 0.0)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, A):
        h = x
        for layer in self.layers:
            h_new = layer(h, A)
            h = torch.clamp(
                h + h_new if h_new.shape == h.shape else h_new, -10, 10
            )
        return h


class MultiScaleGCN(nn.Module):
    """Applies one GCN per spatial scale, then learns a per-spot scale mixture.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output embedding dimension.
    num_layers : int
        Number of GCN layers per scale.
    dropout : float
        Dropout rate.
    n_scales : int
        Number of spatial scales (adjacency matrices).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2, n_scales=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualGCN(in_dim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
                for _ in range(n_scales)
            ]
        )
        self.attn = nn.Sequential(
            nn.Linear(out_dim * n_scales, n_scales), nn.Softmax(dim=-1)
        )

    def forward(self, x, A_list):
        feats = [blk(x, A) for blk, A in zip(self.blocks, A_list)]
        w = self.attn(torch.cat(feats, dim=1))
        out = sum(w[:, i : i + 1] * f for i, f in enumerate(feats))
        return out, w
