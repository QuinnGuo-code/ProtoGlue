"""Decoder and clustering head modules for ProtoGlue."""

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """Per-spot MLP decoder (no graph pass, no additional spatial smoothing).

    Parameters
    ----------
    z_dim : int
        Latent embedding dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Reconstructed feature dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, z_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, A_sp_list=None):
        return self.net(z), None


class DECHead(nn.Module):
    """Deep Embedded Clustering (DEC) soft assignment head.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    d : int
        Embedding dimension.
    alpha : float
        Student-t distribution degrees of freedom.
    """

    def __init__(self, n_clusters, d, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.mu = nn.Parameter(torch.randn(n_clusters, d) * 0.01)

    def soft_assign(self, z):
        """Compute soft cluster assignment probabilities.

        Parameters
        ----------
        z : Tensor, shape (n_spots, d)
            Latent embeddings.

        Returns
        -------
        q : Tensor, shape (n_spots, n_clusters)
            Soft assignment probabilities.
        """
        dist = torch.sum((z[:, None, :] - self.mu[None, :, :]) ** 2, dim=2)
        q = (1.0 + dist / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        return q / (torch.sum(q, dim=1, keepdim=True) + 1e-12)

    @staticmethod
    def target_distribution(q):
        """Compute auxiliary target distribution for DEC.

        Parameters
        ----------
        q : Tensor, shape (n_spots, n_clusters)
            Soft assignment probabilities.

        Returns
        -------
        p : Tensor, shape (n_spots, n_clusters)
            Target distribution.
        """
        w = (q ** 2) / (torch.sum(q, dim=0, keepdim=True) + 1e-12)
        return w / (torch.sum(w, dim=1, keepdim=True) + 1e-12)
