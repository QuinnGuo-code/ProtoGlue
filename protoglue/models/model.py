"""
ProtoGlue: Asymmetric Encode-Decode with Cross-Reconstruction Correspondence.

Forward path
------------
x1 --(MultiScaleGCN + ResGCN)--> z1_sp, z1_fe --(TwoTokenGatedMHA)--> y1 --proj--> y1
x2 --(MultiScaleGCN + ResGCN)--> z2_sp, z2_fe --(TwoTokenGatedMHA)--> y2 --proj--> y2
(y1, y2) --(CrossAttentionGatedFusion)--> z
z --(dec1, dec2)--> x1_hat, x2_hat          [self-reconstruction]
y1 --(dec2)--> x2_cross_hat                 [cross-recon correspondence]
y2 --(dec1)--> x1_cross_hat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from protoglue.config import (
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_HEADS,
    LATENT_DIM_RNA, LATENT_DIM_OM2, FUSION_DIM,
    USE_PROJECTOR, INTER_FUSION, CORRESPONDENCE_TYPE,
    USE_ADAPTIVE_LOSS, ADAPTIVE_LOSS_TERMS,
)
from .layers import ModalityProjector
from .encoders import MultiScaleGCN, ResidualGCN
from .attention import TwoTokenGatedMHA, CrossAttentionGatedFusion
from .decoders import MLPDecoder
from protoglue.losses.total import AdaptiveLossWeights


class ProtoGlue(nn.Module):
    """ProtoGlue multi-omics integration model.

    Parameters
    ----------
    in1 : int
        Input dimension for modality 1 (RNA, typically n_pca).
    in2 : int
        Input dimension for modality 2 (ADT/ATAC, typically n_pca).
    n_scales : int
        Number of spatial graph scales.

    Examples
    --------
    >>> model = ProtoGlue(in1=50, in2=50, n_scales=3)
    >>> out = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t)
    >>> z = out["z"]  # fused latent embedding
    """

    def __init__(self, in1, in2, n_scales: int):
        super().__init__()
        hd = HIDDEN_DIM
        nl, dp, nh = NUM_LAYERS, DROPOUT, NUM_HEADS
        z1, z2, zf = int(LATENT_DIM_RNA), int(LATENT_DIM_OM2), int(FUSION_DIM)

        # Encoders
        self.enc1_sp = MultiScaleGCN(in1, hd, z1, num_layers=nl, dropout=dp, n_scales=n_scales)
        self.enc2_sp = MultiScaleGCN(in2, hd, z2, num_layers=nl, dropout=dp, n_scales=n_scales)
        self.enc1_fe = ResidualGCN(in1, hd, z1, num_layers=nl, dropout=dp)
        self.enc2_fe = ResidualGCN(in2, hd, z2, num_layers=nl, dropout=dp)

        # Intra-modality fusion (spatial emb + feature graph emb)
        self.intra1 = TwoTokenGatedMHA(z1, num_heads=nh, dropout=dp)
        self.intra2 = TwoTokenGatedMHA(z2, num_heads=nh, dropout=dp)

        # Projectors
        self.proj1 = ModalityProjector(z1, zf, dropout=dp) if USE_PROJECTOR else nn.Identity()
        self.proj2 = ModalityProjector(z2, zf, dropout=dp) if USE_PROJECTOR else nn.Identity()

        # Inter-modality fusion
        if str(INTER_FUSION).lower() == "cross_attn":
            self.inter = CrossAttentionGatedFusion(zf, num_heads=nh, dropout=dp)
        else:
            self.inter = TwoTokenGatedMHA(zf, num_heads=nh, dropout=dp)

        # MLP Decoders
        self.dec1 = MLPDecoder(zf, hd, in1, dropout=dp)
        self.dec2 = MLPDecoder(zf, hd, in2, dropout=dp)

        # Adaptive loss weighter
        self.loss_weighter = (
            AdaptiveLossWeights(ADAPTIVE_LOSS_TERMS) if USE_ADAPTIVE_LOSS else None
        )

    def forward(self, x1, x2, A_sp_list, A_f1, A_f2, dec_head=None):
        """Forward pass.

        Parameters
        ----------
        x1 : Tensor, shape (n_spots, in1)
            Modality 1 PCA embeddings.
        x2 : Tensor, shape (n_spots, in2)
            Modality 2 PCA embeddings.
        A_sp_list : list of sparse Tensor
            Multi-scale spatial adjacency matrices.
        A_f1 : sparse Tensor
            Feature-space graph for modality 1.
        A_f2 : sparse Tensor
            Feature-space graph for modality 2.
        dec_head : DECHead, optional
            Clustering head for soft assignment.

        Returns
        -------
        dict
            Dictionary containing embeddings, reconstructions, and attention weights.
        """
        # Encode
        z1_sp, w1 = self.enc1_sp(x1, A_sp_list)
        z2_sp, w2 = self.enc2_sp(x2, A_sp_list)
        z1_fe = self.enc1_fe(x1, A_f1)
        z2_fe = self.enc2_fe(x2, A_f2)

        # Intra-modality fusion
        y1_raw, a1 = self.intra1(z1_sp, z1_fe)
        y2_raw, a2 = self.intra2(z2_sp, z2_fe)
        y1 = self.proj1(y1_raw)
        y2 = self.proj2(y2_raw)

        # Inter-modality fusion -> fused embedding z
        z, a = self.inter(y1, y2)

        # Self-reconstruction
        x1_hat, wdec1 = self.dec1(z, A_sp_list)
        x2_hat, wdec2 = self.dec2(z, A_sp_list)

        # Cross-reconstruction correspondence
        if CORRESPONDENCE_TYPE == "cross_recon":
            y1_hat, _ = self.dec2(y1, A_sp_list)
            y2_hat, _ = self.dec1(y2, A_sp_list)
        else:
            # Fallback: GCN cycle (v3 style)
            x2_from_y1, _ = self.dec2(y1, A_sp_list)
            y1_hat = self.proj2(self.enc2_sp(x2_from_y1, A_sp_list)[0])
            x1_from_y2, _ = self.dec1(y2, A_sp_list)
            y2_hat = self.proj1(self.enc1_sp(x1_from_y2, A_sp_list)[0])

        q = dec_head.soft_assign(z) if dec_head is not None else None

        return {
            "z": z,
            "x1_hat": x1_hat,
            "x2_hat": x2_hat,
            "y1": y1,
            "y2": y2,
            "y1_hat": y1_hat,
            "y2_hat": y2_hat,
            "alpha_modality": a,
            "alpha_rna_intra": a1,
            "alpha_omics2_intra": a2,
            "w_scale_rna": w1,
            "w_scale_omics2": w2,
            "w_dec1": wdec1,
            "w_dec2": wdec2,
            "q": q,
        }
