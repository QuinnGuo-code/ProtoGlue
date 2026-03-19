"""
ProtoGlue default configuration.

All hyperparameters are defined here with sensible defaults matching
the human lymph node benchmark. Users can override them before model
construction or pass them as function arguments where supported.
"""

import numpy as np

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SEED = 2022

# ── Preprocessing ──────────────────────────────────────────────
N_HVG = 3000          # number of highly variable genes for RNA
N_PCA = 50            # PCA target dimensionality

# ── Graph construction ─────────────────────────────────────────
USE_AMSG = True                # use adaptive multi-scale spatial graph
K_BASE = 5                     # base k for adaptive KNN
K_MIN = 3                      # minimum neighbors
K_MAX = 10                     # maximum neighbors
DENSITY_ALPHA = 0.2            # density-adaptive exponent
SPATIAL_SCALES = [3, 8]        # fixed spatial KNN scales
FEAT_K = 15                    # feature graph neighbor count

# ── Model architecture ─────────────────────────────────────────
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_HEADS = 8

LATENT_DIM_RNA = LATENT_DIM
LATENT_DIM_OM2 = LATENT_DIM
FUSION_DIM = LATENT_DIM

USE_PROJECTOR = True
INTER_FUSION = "cross_attn"    # "cross_attn" or "two_token"

TEMP_START = 1.0               # temperature annealing start
TEMP_END = 0.25                # temperature annealing end
TEMP_ANNEAL_EPOCHS = 200       # epochs over which to anneal temperature

DECODER_TYPE = "mlp"           # decoder type
CORRESPONDENCE_TYPE = "cross_recon"  # "cross_recon" or "cycle"

# ── VICReg coefficients ────────────────────────────────────────
VICREG_SIM_COEFF = 1.0
VICREG_VAR_COEFF = 1.0
VICREG_COV_COEFF = 0.5

# ── Adaptive loss weights ──────────────────────────────────────
USE_ADAPTIVE_LOSS = True

BASE_LOSS_WEIGHTS = {
    "recon":        1.0,
    "corr":         0.8,
    "smooth":       0.02,
    "vicreg":       0.05,
    "cluster":      3.0,
    "balance":      1.5,
    "alpha_ent":    0.05,
    "alpha_var":    0.03,
    "alpha_smooth": 0.01,
}
ADAPTIVE_LOSS_TERMS = [
    "recon", "corr", "smooth", "vicreg",
    "cluster", "balance",
    "alpha_ent", "alpha_var", "alpha_smooth",
]

# ── Training ───────────────────────────────────────────────────
LR = 5e-4
WEIGHT_DECAY = 1e-4
PRETRAIN_EPOCHS = 300
FINETUNE_EPOCHS = 600
DEC_UPDATE_INTERVAL = 5
WARMUP_EPOCHS = 30
PATIENCE = 150
GRAD_CLIP = 3.0
USE_LR_PLATEAU = True
USE_AMP = True

LAM_RECON = 1.0
LAM_CORR = 0.20
LAM_SMOOTH = 0.01
LAM_VICREG = 0.03
LAM_CLUSTER = 2.0
LAM_BALANCE = 0.30
LAM_ALPHA_ENT = 0.05
LAM_ALPHA_VAR = 0.01
LAM_ALPHA_SMOOTH = 0.005

P_SMOOTH = 0.0                 # spatial smoothing of DEC target distribution

# ── Clustering ─────────────────────────────────────────────────
AUTO_SELECT_K = True
N_CLUSTERS = 6
K_GRID = list(range(5, 9))
K_SCAN_SAMPLE = 2000
K_SCAN_STABILITY_RUNS = 3
K_SCORE_W_SIL = 1.0
K_SCORE_W_PURITY = 1.0
K_SCORE_W_MORAN = 0.5
K_SCORE_W_STAB = 0.5
USE_GT_FOR_K_SELECTION = False
K_SCORE_W_GT = 3.0
K_MIN_SELECT = 5
K_SELECT_W_ELBOW = 0.6
K_SELECT_W_LOGK = 0.8

MIN_CLUSTER_FRAC = 0.01
COLLAPSE_CHECK_INTERVAL = 10

MARKER_NGENES = 5
HEATMAP_CLIP = 2.0

REFINE_TOPK = False
TOPK_CANDIDATES = 3
SHORT_FINETUNE_EPOCHS = 40

CLUSTER_METHODS = ["kmeans", "gmm", "leiden"]
FORCE_K = None
PICK_BEST_BY_GT = False
LEIDEN_N_NEIGHBORS = 15
REFINE_LARGE_CLUSTERS = False
REFINE_SIZE_THRESH = 0.10
REFINE_MAX_SUBCLUSTERS = 3
