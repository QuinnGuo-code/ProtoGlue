"""
High-level API for end-to-end ProtoGlue analysis.

Provides :func:`run_pipeline` which orchestrates data loading, graph
construction, model training, K selection, fine-tuning, clustering,
and evaluation in a single call.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
from sklearn.cluster import KMeans

import protoglue.config as default_config
from protoglue.data.utils import fix_seed
from protoglue.data.preprocessing import load_and_preprocess
from protoglue.data.graph import build_graphs, graphs_to_torch
from protoglue.models.model import ProtoGlue
from protoglue.models.decoders import DECHead
from protoglue.training.trainer import run_stage
from protoglue.training.k_selection import scan_select_k, eval_labels
from protoglue.losses.total import total_loss
from protoglue.losses.smooth import calibrate_lam_smooth


def run_pipeline(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    rna_file: Optional[str] = None,
    om2_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    device: str = None,
    seed: int = 2022,
    auto_k: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete ProtoGlue analysis pipeline.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .h5ad files and optional annotation.csv.
    output_dir : str or Path
        Output directory for checkpoints, figures, etc.
    rna_file, om2_file : str, optional
        Explicit filenames for RNA and second modality. Auto-detected if None.
    config : dict, optional
        Override default hyperparameters (key names must match ``protoglue.config``).
    device : str, optional
        Compute device (``"cuda"`` or ``"cpu"``). Auto-detected if None.
    seed : int
        Random seed for reproducibility.
    auto_k : bool
        Whether to automatically select the number of clusters K.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Result dictionary containing:
        - ``"latent"``: final fused embedding (ndarray)
        - ``"labels"``: cluster labels (ndarray)
        - ``"metrics"``: evaluation metrics (dict)
        - ``"adata_rna"``: annotated RNA AnnData
        - ``"model"``: trained ProtoGlue model
        - ``"dec_head"``: trained DEC clustering head

    Examples
    --------
    >>> import protoglue as pg
    >>> result = pg.run_pipeline("data/human_lymph_node", "output/lymph_node")
    >>> print(result["metrics"])
    >>> labels = result["labels"]
    """
    # 0. Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge user config overrides
    cfg = default_config
    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                print(f"Warning: unknown config key '{key}', skipping.")

    fix_seed(seed)

    # 1. Load and preprocess data
    if verbose:
        print("=" * 60)
        print("Step 1/6: Loading and preprocessing data...")
    pca1, pca2, coords, adata_rna, adata_om2 = load_and_preprocess(
        data_dir=data_dir,
        rna_file=rna_file,
        om2_file=om2_file,
        n_hvg=cfg.N_HVG,
        n_pca=cfg.N_PCA,
        seed=seed,
        verbose=verbose,
    )

    # 2. Build graphs
    if verbose:
        print("=" * 60)
        print("Step 2/6: Building graphs...")
    A_sp_list, A_f1, A_f2 = build_graphs(
        coords=coords,
        pca1=pca1,
        pca2=pca2,
        use_amsg=cfg.USE_AMSG,
        k_base=cfg.K_BASE,
        k_min=cfg.K_MIN,
        k_max=cfg.K_MAX,
        density_alpha=cfg.DENSITY_ALPHA,
        spatial_scales=cfg.SPATIAL_SCALES,
        feat_k=cfg.FEAT_K,
    )
    dev = torch.device(device)
    A_sp_t, A_f1_t, A_f2_t = graphs_to_torch(A_sp_list, A_f1, A_f2, device=dev)
    X1_t = torch.tensor(pca1, dtype=torch.float32, device=dev)
    X2_t = torch.tensor(pca2, dtype=torch.float32, device=dev)

    # 3. Initialize model
    if verbose:
        print("=" * 60)
        print("Step 3/6: Initializing model...")
    model = ProtoGlue(
        in1=X1_t.shape[1], in2=X2_t.shape[1], n_scales=len(A_sp_t)
    ).to(dev)

    # Calibrate smoothness weight
    lam_smooth = calibrate_lam_smooth(model, X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t)
    cfg.LAM_SMOOTH = lam_smooth
    if verbose:
        print(f"Calibrated LAM_SMOOTH = {lam_smooth}")

    # 4. Pretrain
    if verbose:
        print("=" * 60)
        print("Step 4/6: Pretraining...")
    optimizer_pre = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler_pre = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pre, mode="min", factor=0.5, patience=40, min_lr=1e-6
        )
        if cfg.USE_LR_PLATEAU
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP and "cuda" in device)

    pretrain_logs = run_stage(
        stage="pretrain",
        epochs=cfg.PRETRAIN_EPOCHS,
        model=model,
        X1_t=X1_t, X2_t=X2_t,
        A_sp_t=A_sp_t, A_f1_t=A_f1_t, A_f2_t=A_f2_t,
        optimizer=optimizer_pre,
        scheduler=scheduler_pre,
        scaler=scaler,
        loss_fn=total_loss,
        output_dir=output_dir,
        ckpt_tag="pretrain",
        WARMUP_EPOCHS=cfg.WARMUP_EPOCHS,
        PATIENCE=cfg.PATIENCE,
        GRAD_CLIP=cfg.GRAD_CLIP,
        SEED=seed,
    )

    # Load best pretrain checkpoint
    ckpt_path = output_dir / "best_pretrain.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])

    # 5. K selection
    if verbose:
        print("=" * 60)
        print("Step 5/6: Selecting optimal K...")
    model.eval()
    with torch.no_grad():
        out0 = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t)
    z0 = out0["z"].detach().cpu().numpy()

    gt = None
    if "manual_anno" in adata_rna.obs.columns:
        gt_raw = adata_rna.obs["manual_anno"]
        if gt_raw.notna().sum() > 0:
            gt = gt_raw.values

    if auto_k:
        best_k, kscan_df = scan_select_k(
            z=z0,
            A_sp0=A_sp_list[0],
            gt=gt,
            K_GRID=cfg.K_GRID,
            SEED=seed,
        )
        cfg.N_CLUSTERS = best_k
        if verbose:
            print(f"Selected K = {best_k}")
            print(kscan_df[["K", "silhouette", "spatial_edge_purity", "score_select"]].to_string(index=False))
    else:
        best_k = cfg.N_CLUSTERS

    # 6. Fine-tune with clustering
    if verbose:
        print("=" * 60)
        print("Step 6/6: Fine-tuning with DEC clustering...")
    dec_head = DECHead(best_k, cfg.FUSION_DIM).to(dev)

    # Initialize prototypes with KMeans
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=50).fit(z0)
    with torch.no_grad():
        dec_head.mu.copy_(
            torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
        )

    optimizer_ft = torch.optim.AdamW(
        list(model.parameters()) + list(dec_head.parameters()),
        lr=cfg.LR * 0.5,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler_ft = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, mode="min", factor=0.5, patience=40, min_lr=1e-6
        )
        if cfg.USE_LR_PLATEAU
        else None
    )
    scaler_ft = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP and "cuda" in device)

    finetune_logs = run_stage(
        stage="finetune",
        epochs=cfg.FINETUNE_EPOCHS,
        model=model,
        X1_t=X1_t, X2_t=X2_t,
        A_sp_t=A_sp_t, A_f1_t=A_f1_t, A_f2_t=A_f2_t,
        dec_head=dec_head,
        optimizer=optimizer_ft,
        scheduler=scheduler_ft,
        scaler=scaler_ft,
        loss_fn=total_loss,
        warm_offset=cfg.PRETRAIN_EPOCHS,
        output_dir=output_dir,
        ckpt_tag="finetune",
        WARMUP_EPOCHS=cfg.WARMUP_EPOCHS,
        PATIENCE=cfg.PATIENCE,
        GRAD_CLIP=cfg.GRAD_CLIP,
        DEC_UPDATE_INTERVAL=cfg.DEC_UPDATE_INTERVAL,
        P_SMOOTH=cfg.P_SMOOTH,
        COLLAPSE_CHECK_INTERVAL=cfg.COLLAPSE_CHECK_INTERVAL,
        MIN_CLUSTER_FRAC=cfg.MIN_CLUSTER_FRAC,
        SEED=seed,
        TEMP_ANNEAL_EPOCHS=cfg.TEMP_ANNEAL_EPOCHS,
        TEMP_START=cfg.TEMP_START,
        TEMP_END=cfg.TEMP_END,
    )

    # Load best finetune checkpoint
    ckpt_path = output_dir / "best_finetune.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if ckpt["dec_head"] is not None:
            dec_head.load_state_dict(ckpt["dec_head"])

    # Final evaluation
    if verbose:
        print("=" * 60)
        print("Generating final results...")
    model.eval()
    with torch.no_grad():
        out_final = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, dec_head=dec_head)
    z_final = out_final["z"].detach().cpu().numpy()
    q = out_final["q"].detach().cpu().numpy()
    labels = q.argmax(axis=1).astype(int)

    metrics = eval_labels(
        labels,
        A_sp0=A_sp_list[0],
        gt=gt,
        z=z_final,
        x1=pca1,
        x2=pca2,
    )

    if verbose:
        print("=" * 60)
        print("Results:")
        print(f"  Clusters: {metrics.get('n_clusters', '?')}")
        print(f"  Spatial edge purity: {metrics.get('spatial_edge_purity', 'N/A'):.4f}")
        if "ARI" in metrics and np.isfinite(metrics["ARI"]):
            print(f"  ARI: {metrics['ARI']:.4f}")
            print(f"  NMI: {metrics['NMI']:.4f}")

    # Store labels in AnnData
    adata_rna.obs["protoglue_labels"] = labels.astype(str)
    adata_rna.obsm["protoglue_latent"] = z_final

    return {
        "latent": z_final,
        "labels": labels,
        "q": q,
        "metrics": metrics,
        "adata_rna": adata_rna,
        "model": model,
        "dec_head": dec_head,
        "pretrain_logs": pretrain_logs,
        "finetune_logs": finetune_logs,
    }
