import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import seaborn as sns
from pathlib import Path
import warnings

# Import internal helper modules
from .theme import *   # apply theme settings
from .colors import cluster_cmap, PALETTE_DIVERGE, PALETTE_SEQUENTIAL
from .utils import save_fig_pub, despine, make_colorbar, add_scalebar, _to_1d

def plot_overview(
    xy_plot: np.ndarray,
    labels: np.ndarray,
    umap_xy: np.ndarray,
    w_mod: np.ndarray,
    n_clusters: int,
    sil_val: float,
    ari_val: float,
    nmi_val: float,
    edge_purity: float,
    s_size: float = None,
    _colors: list = None,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 1: Overview (spatial clusters + UMAP + modality weights)"""
    if _colors is None:
        _colors = cluster_cmap(n_clusters)
    _c_arr = np.array([_colors[int(l)] for l in labels])

    fig = plt.figure(figsize=(7.2, 2.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35, left=0.04, right=0.96, top=0.88, bottom=0.12)

    # (a) Spatial clusters
    ax_sp = fig.add_subplot(gs[0])
    for k in range(n_clusters):
        idx = labels == k
        ax_sp.scatter(xy_plot[idx, 0], xy_plot[idx, 1],
                      s=s_size, color=_colors[k], linewidths=0,
                      label=f'C{k}', rasterized=True, zorder=2)
    ax_sp.axis('off')
    ax_sp.set_title('Spatial domains', fontweight='bold', pad=5)
    if n_clusters <= 12:
        leg = ax_sp.legend(loc='lower left', markerscale=2.0, ncol=2,
                           columnspacing=0.4, handletextpad=0.2,
                           bbox_to_anchor=(0, -0.02), frameon=False, fontsize=5)
        for lh in leg.legend_handles:
            lh.set_sizes([18])

    # (b) UMAP
    ax_um = fig.add_subplot(gs[1])
    order = np.random.permutation(len(labels))
    ax_um.scatter(umap_xy[order, 0], umap_xy[order, 1],
                  s=s_size*0.7, c=np.array(_c_arr)[order], linewidths=0,
                  alpha=0.8, rasterized=True)
    ax_um.set_xlabel('UMAP 1')
    ax_um.set_ylabel('UMAP 2')
    ax_um.set_title('UMAP embedding', fontweight='bold', pad=5)
    despine(ax_um)
    for k in range(n_clusters):
        idx = labels == k
        cx, cy = np.median(umap_xy[idx, 0]), np.median(umap_xy[idx, 1])
        ax_um.annotate(str(k), (cx, cy), fontsize=5.5, ha='center', va='center',
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='circle,pad=0.15', fc=_colors[k],
                                 ec='white', lw=0.3, alpha=0.92))

    # (c) Modality weight
    ax_mw = fig.add_subplot(gs[2])
    dev = w_mod - 0.5
    abs_dev = np.abs(dev)
    v = float(np.percentile(abs_dev, 98))
    v = max(v, 1e-3)
    norm_dev = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    sc_mw = ax_mw.scatter(xy_plot[:, 0], xy_plot[:, 1],
                          s=s_size, c=dev,
                          cmap='seismic', norm=norm_dev,
                          linewidths=0, alpha=1.0, rasterized=True)
    ax_mw.axis('off')
    ax_mw.set_title('Modality deviation (RNA − Protein)', fontweight='bold', pad=5)
    cb = plt.colorbar(sc_mw, ax=ax_mw, fraction=0.04, pad=0.02, shrink=0.78, aspect=20)
    cb.set_label('Protein-dominant (−) ← deviation → RNA-dominant (+)', fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.3, length=2)
    cb.outline.set_linewidth(0.3)

    # Metric annotation
    if np.isfinite(sil_val):
        sil_text = f'Sil = {sil_val:.3f}'
    else:
        sil_text = 'Sil = nan'
    metrics_note = sil_text + f'\nPurity = {edge_purity:.3f}'
    if not np.isnan(ari_val):
        metrics_note += f'\nARI = {ari_val:.3f}\nNMI = {nmi_val:.3f}'
    ax_mw.text(0.02, 0.05, metrics_note,
               transform=ax_mw.transAxes, fontsize=5.5, va='bottom', color='#333333',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc', lw=0.3, alpha=0.92))

    # Save
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "overview_spatial_umap_weight.png")

    return fig


def plot_multiscale_attention(
    xy_plot: np.ndarray,
    w_scale_rna: np.ndarray,
    w_scale_om2: np.ndarray,
    n_scales: int,
    spatial_scales: list,          # additional parameter
    s_size: float = None,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 2: Multi-scale spatial attention weights"""
    # Generate labels from spatial_scales parameter
    scale_labels = (["AMSG"] + [f"k={k}" for k in spatial_scales])[:n_scales]

    fig, axes = plt.subplots(2, n_scales, figsize=(2.2 * n_scales, 4.4),
                              gridspec_kw={'hspace': 0.22, 'wspace': 0.08})
    if n_scales == 1:
        axes = axes.reshape(2, 1)

    SWAP_RNA_PROTEIN_ROWS = False  # set to True if needed

    for col, (wRNA, wOM2, slabel) in enumerate(zip(w_scale_rna.T, w_scale_om2.T, scale_labels)):
        pairs = [(wRNA, "RNA"), (wOM2, "Protein")]
        if SWAP_RNA_PROTEIN_ROWS:
            pairs = list(reversed(pairs))
        for row, (w, modal_label) in enumerate(pairs):
            ax = axes[row, col]
            vmin, vmax = np.percentile(w, 2), np.percentile(w, 98)
            sc = ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=s_size * 0.8, c=w,
                            cmap=PALETTE_SEQUENTIAL, vmin=vmin, vmax=vmax,
                            linewidths=0, rasterized=True)
            ax.axis('off')
            if row == 0:
                ax.set_title(slabel, fontsize=7, fontweight='bold', pad=3)
            if col == 0:
                ax.text(-0.05, 0.5, modal_label, transform=ax.transAxes,
                        fontsize=7, fontweight='bold', ha='right', va='center', rotation=90)
            if col == n_scales - 1:
                cb = plt.colorbar(sc, ax=ax, fraction=0.06, pad=0.02, shrink=0.82)
                cb.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
                cb.outline.set_linewidth(0.3)

    fig.suptitle('Multi-scale spatial attention weights', fontsize=8.5, fontweight='bold', y=0.98)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "multiscale_spatial_attention.png")

    return fig


def plot_k_scan(
    kscan_df_plot: pd.DataFrame,
    N_CLUSTERS_SEL: int,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 3: K-scan scoring curves"""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5),
                              gridspec_kw={'wspace': 0.45, 'left': 0.09, 'right': 0.92, 'top': 0.86, 'bottom': 0.16})

    ks = kscan_df_plot["K"].to_numpy()
    K_ELBOW = kscan_df_plot["K_elbow"].iloc[0] if "K_elbow" in kscan_df_plot.columns else ks[len(ks) // 2]

    # (a) Individual metrics
    ax = axes[0]
    metric_cfg = [
        ("silhouette", "#E64B35", "o", "Silhouette"),
        ("spatial_edge_purity", "#4DBBD5", "s", "Spatial purity"),
        ("moran_median", "#00A087", "^", "Moran's I"),
        ("stability_ari", "#3C5488", "D", "Stability (ARI)")
    ]
    for col_name, color, marker, lab in metric_cfg:
        if col_name in kscan_df_plot.columns and kscan_df_plot[col_name].notna().any():
            vals = kscan_df_plot[col_name].to_numpy(dtype=float)
            vals_z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-9)
            ax.plot(ks, vals_z, color=color, lw=1.0, marker=marker, ms=3.5,
                    markeredgecolor='white', markeredgewidth=0.3,
                    label=lab, zorder=3)

    ax.axvline(int(K_ELBOW), ls='--', lw=0.7, color='#999999', alpha=0.7,
               label=f'Elbow K={int(K_ELBOW)}')
    ax.axvline(int(N_CLUSTERS_SEL), ls=':', lw=1.0, color='#E64B35',
               label=f'Selected K={int(N_CLUSTERS_SEL)}')
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Z-score')
    ax.set_title('Cluster quality metrics', fontweight='bold', pad=5)
    ax.legend(loc='upper right', ncol=1, handlelength=1.0, fontsize=5.5)
    despine(ax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # (b) Composite score
    ax = axes[1]
    if "score_select" in kscan_df_plot.columns:
        ax.plot(ks, kscan_df_plot["score_select"].to_numpy(), color='#3C5488',
                lw=1.2, marker='o', ms=3.5, markeredgecolor='white', markeredgewidth=0.3,
                label='Selection score', zorder=4)
        ax.plot(ks, kscan_df_plot["score"].to_numpy(), color='#8491B4',
                lw=0.8, marker='s', ms=2.5, ls='--', label='Base score', zorder=3)

        ax2 = ax.twinx()
        ax2.plot(ks, kscan_df_plot["inertia"].to_numpy(), color='#cccccc', lw=0.8,
                 ls='-.', marker='^', ms=2, alpha=0.7, label='Inertia')
        ax2.set_ylabel('Inertia', fontsize=6, color='#aaaaaa')
        ax2.tick_params(axis='y', labelsize=5, colors='#aaaaaa')
        ax2.spines['right'].set_edgecolor('#cccccc')
        ax2.spines['right'].set_linewidth(0.4)
        ax2.spines['top'].set_visible(False)

    ax.axvline(int(K_ELBOW), ls='--', lw=0.7, color='#999999', alpha=0.7,
               label=f'Elbow K={int(K_ELBOW)}')
    ax.axvline(int(N_CLUSTERS_SEL), ls=':', lw=1.0, color='#E64B35',
               label=f'Selected K={int(N_CLUSTERS_SEL)}')
    best_row_v = kscan_df_plot[kscan_df_plot["K"] == int(N_CLUSTERS_SEL)]
    if len(best_row_v) > 0 and "score_select" in kscan_df_plot.columns:
        bv = float(best_row_v["score_select"].values[0])
        ax.scatter([int(N_CLUSTERS_SEL)], [bv], s=55, zorder=6,
                   color='#E64B35', edgecolors='white', linewidths=0.8)
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Composite score')
    ax.set_title('K selection (elbow-guided)', fontweight='bold', pad=5)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels() if 'ax2' in locals() else ([], [])
    ax.legend(lines1 + lines2, labs1 + labs2, loc='upper right', ncol=1,
              handlelength=1.0, fontsize=5.5)
    despine(ax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "k_scan.png")

    return fig


def plot_train_loss(
    log_df_plot: pd.DataFrame,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 4: Training curves (loss decomposition)"""
    loss_cols = [c for c in ["loss", "recon", "corr", "smooth", "vicreg", "cluster", "balance"]
                 if c in log_df_plot.columns]
    loss_labels = {
        "loss": "Total loss", "recon": "Reconstruction", "corr": "Correspondence",
        "smooth": "Laplacian smooth", "vicreg": "VICReg", "cluster": "DEC-KL",
        "balance": "Balance"
    }
    loss_colors = {
        "loss": "#333333", "recon": "#E64B35", "corr": "#4DBBD5",
        "smooth": "#00A087", "vicreg": "#F39B7F", "cluster": "#3C5488",
        "balance": "#8491B4"
    }
    stages = log_df_plot["stage"].unique().tolist()
    n_cols = min(len(loss_cols), 4)
    n_rows = int(np.ceil(len(loss_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                              gridspec_kw={'hspace': 0.55, 'wspace': 0.40,
                                           'left': 0.08, 'right': 0.97, 'top': 0.90, 'bottom': 0.10})
    axes = np.array(axes).flatten()
    stage_colors = {"pretrain": "#4DBBD5", "finetune": "#E64B35"}

    for i, col in enumerate(loss_cols):
        ax = axes[i]
        for st in stages:
            sub = log_df_plot[log_df_plot["stage"] == st].copy()
            ep = sub["epoch"].to_numpy() if "epoch" in sub.columns else sub.index.to_numpy()
            vals = sub[col].to_numpy(dtype=float)
            c = stage_colors.get(st, "#888888")
            ax.plot(ep, vals, color=c, lw=0.8, alpha=0.95, label=st)
        ax.set_title(loss_labels.get(col, col), fontsize=6.5, fontweight='bold', pad=3)
        ax.set_xlabel('Epoch', fontsize=5.5)
        ax.set_ylabel('Loss', fontsize=5.5)
        despine(ax)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
        if i == 0:
            ax.legend(loc='upper right', fontsize=5)

    for j in range(len(loss_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Training loss decomposition', fontsize=8.5, fontweight='bold', y=0.98)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "train_loss_decompose.png")

    return fig


def plot_marker_heatmap(
    adata_out,
    cluster_key: str,
    marker_genes: list,
    use_raw_marker: bool,
    HEATMAP_CLIP: float,
    _colors: list,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 5: Marker gene heatmap"""
    if len(marker_genes) == 0:
        return None

    ad_view = adata_out.raw.to_adata() if (use_raw_marker and adata_out.raw is not None) else adata_out
    genes = [g for g in marker_genes if g in ad_view.var_names]
    if len(genes) == 0:
        return None

    grp = adata_out.obs[cluster_key].astype(str).to_numpy()
    order = sorted(np.unique(grp), key=lambda x: int(x) if x.isdigit() else x)
    K_ = len(order)

    X_ = ad_view[:, genes].X
    X_ = X_.toarray() if sp.issparse(X_) else np.asarray(X_)
    M = np.vstack([X_[grp == k].mean(0) if (grp == k).sum() > 0 else np.zeros(len(genes)) for k in order])
    mu_ = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True) + 1e-8
    Z_ = np.clip((M - mu_) / sd, -HEATMAP_CLIP, HEATMAP_CLIP)

    G = len(genes)
    cell_w = max(0.28, 5.0 / G)
    cell_h = max(0.30, 3.5 / K_)
    fig_w = min(12, G * cell_w + 2.0)
    fig_h = min(8, K_ * cell_h + 1.8)

    fig, (ax_hm, ax_bar) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [K_, 0.5], 'hspace': 0.03,
                     'left': 0.12, 'right': 0.88, 'top': 0.92, 'bottom': 0.18}
    )

    cmap = plt.cm.get_cmap(PALETTE_DIVERGE)
    im = ax_hm.imshow(Z_, aspect='auto', cmap=cmap, vmin=-HEATMAP_CLIP, vmax=HEATMAP_CLIP, interpolation='nearest')
    ax_hm.set_yticks(range(K_))
    ax_hm.set_yticklabels([f'C{k}' for k in order], fontsize=6)
    ax_hm.set_xticks([])
    ax_hm.tick_params(axis='y', length=0)

    # colorbar
    pos_hm = ax_hm.get_position()
    cax = fig.add_axes([pos_hm.x1 + 0.015, pos_hm.y0 + pos_hm.height * 0.2,
                        0.012, pos_hm.height * 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('z-score', fontsize=5.5)
    cb.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
    cb.outline.set_linewidth(0.3)

    # Left cluster colorbar
    left_ax = fig.add_axes([pos_hm.x0 - 0.025, pos_hm.y0, 0.015, pos_hm.height])
    bar_colors = np.array([[mcolors.to_rgb(_colors[i % len(_colors)])] for i in range(K_)])
    left_ax.imshow(bar_colors, aspect='auto', interpolation='nearest')
    left_ax.axis('off')

    # Bottom gene group colorbar
    gene_to_cluster = {}
    rg = adata_out.uns.get('rank_genes_groups', {})
    if 'names' in rg:
        rg_names = rg['names']
        grps_names = rg_names.dtype.names if hasattr(rg_names, 'dtype') and rg_names.dtype.names else list(rg_names.keys())
        for g_ in grps_names:
            for gene_ in list(rg_names[g_][:5]):  # MARKER_NGENES defaults to 5
                if str(gene_) not in gene_to_cluster:
                    gene_to_cluster[str(gene_)] = str(g_)
    gene_colors = [mcolors.to_rgb(_colors[int(gene_to_cluster.get(g_, "0")) % len(_colors)])
                   if gene_to_cluster.get(g_) else (0.85, 0.85, 0.85) for g_ in genes]
    gene_colors_arr = np.array([[c] for c in gene_colors]).transpose(1, 0, 2)
    ax_bar.imshow(gene_colors_arr, aspect='auto', interpolation='nearest')
    ax_bar.set_xticks(range(G))
    ax_bar.set_xticklabels(genes, rotation=55, ha='right', fontsize=5)
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    ax_hm.set_title(f'Cluster marker genes (z-score clip ± {HEATMAP_CLIP})', fontsize=8, fontweight='bold', pad=5)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "marker_heatmap.png")

    return fig


def plot_dotplot(
    adata_out,
    cluster_key: str,
    marker_genes: list,
    use_raw_marker: bool,
    HEATMAP_CLIP: float,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 6: Marker gene dot plot"""
    if len(marker_genes) == 0:
        return None

    ad_view = adata_out.raw.to_adata() if (use_raw_marker and adata_out.raw is not None) else adata_out
    genes = [g for g in marker_genes if g in ad_view.var_names]
    if len(genes) == 0:
        return None

    grp = adata_out.obs[cluster_key].astype(str).to_numpy()
    order = sorted(np.unique(grp), key=lambda x: int(x) if x.isdigit() else x)
    K_ = len(order)

    X_ = ad_view[:, genes].X
    X_ = X_.toarray() if sp.issparse(X_) else np.asarray(X_)
    means_, fracs_ = [], []
    for k in order:
        idx = np.where(grp == k)[0]
        if len(idx):
            means_.append(X_[idx].mean(0))
            fracs_.append((X_[idx] > 0).mean(0))
        else:
            means_.append(np.zeros(len(genes)))
            fracs_.append(np.zeros(len(genes)))
    M = np.vstack(means_)
    Fr_ = np.vstack(fracs_)
    mu_ = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True) + 1e-8
    Z_ = np.clip((M - mu_) / sd, -HEATMAP_CLIP, HEATMAP_CLIP)

    G = len(genes)
    fig_w = min(12, G * 0.38 + 1.8)
    fig_h = min(8, K_ * 0.36 + 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h),
                            gridspec_kw={'left': 0.12, 'right': 0.86, 'top': 0.90, 'bottom': 0.22})
    cmap_ = plt.cm.get_cmap(PALETTE_DIVERGE)
    norm_ = mcolors.Normalize(vmin=-HEATMAP_CLIP, vmax=HEATMAP_CLIP)

    for yi, k in enumerate(order):
        for xi, g in enumerate(genes):
            frac = float(Fr_[yi, xi])
            zsval = float(Z_[yi, xi])
            size = max(4, frac * 200)
            color = cmap_(norm_(zsval))
            ax.scatter(xi, yi, s=size, color=color,
                       edgecolors='white', linewidths=0.25, zorder=3)

    for xi in range(G):
        ax.axvline(xi, color='#f0f0f0', lw=0.3, zorder=1)
    for yi in range(K_):
        ax.axhline(yi, color='#f0f0f0', lw=0.3, zorder=1)

    ax.set_xticks(range(G))
    ax.set_xticklabels(genes, rotation=55, ha='right', fontsize=5)
    ax.set_yticks(range(K_))
    ax.set_yticklabels([f'C{k}' for k in order], fontsize=6)

    pos_ = ax.get_position()
    cax = fig.add_axes([pos_.x1 + 0.02, pos_.y0 + pos_.height * 0.15,
                        0.012, pos_.height * 0.7])
    cb_ = ColorbarBase(cax, cmap=cmap_, norm=norm_, orientation='vertical')
    cb_.set_label('z-score', fontsize=5.5)
    cb_.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
    cb_.outline.set_linewidth(0.3)

    size_vals = [0.1, 0.3, 0.6, 1.0]
    leg_handles = [Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='#999999', markeredgecolor='white', markeredgewidth=0.25,
                          markersize=np.sqrt(max(4, v * 200)) / 2,
                          label=f'{int(v * 100)}%') for v in size_vals]
    ax.legend(handles=leg_handles, title='Frac. expr.', title_fontsize=5.5,
              loc='lower left', bbox_to_anchor=(-0.01, -0.42), ncol=len(size_vals),
              handletextpad=0.1, columnspacing=0.5, frameon=False, fontsize=5.5)

    ax.invert_yaxis()
    ax.set_title('Marker gene dot plot', fontsize=8, fontweight='bold', pad=5)
    despine(ax)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "marker_dotplot.png")

    return fig


def plot_umap_density(
    umap_xy: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    _colors: list,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 7: UMAP density plot"""
    from scipy.stats import gaussian_kde

    n_show = n_clusters
    ncols_den = min(n_show, 5)
    nrows_den = int(np.ceil(n_show / ncols_den))

    fig, axes = plt.subplots(nrows_den, ncols_den,
                              figsize=(2.0 * ncols_den, 2.2 * nrows_den),
                              gridspec_kw={'wspace': 0.05, 'hspace': 0.15,
                                           'left': 0.03, 'right': 0.97,
                                           'top': 0.88, 'bottom': 0.04})
    axes = list(np.array(axes).flatten())

    xmin, xmax = umap_xy[:, 0].min() - 0.5, umap_xy[:, 0].max() + 0.5
    ymin, ymax = umap_xy[:, 1].min() - 0.5, umap_xy[:, 1].max() + 0.5
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    pos = np.vstack([xx.ravel(), yy.ravel()])

    for ki in range(n_show):
        ax = axes[ki]
        ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=0.5, color='#e8e8e8',
                   linewidths=0, rasterized=True, zorder=1)
        idx_k = labels == ki
        if idx_k.sum() > 5:
            try:
                kde = gaussian_kde(umap_xy[idx_k].T, bw_method=0.3)
                Z_kde = kde(pos).reshape(xx.shape)
                fade_cmap = LinearSegmentedColormap.from_list('fade', ['#ffffff00', _colors[ki]], N=128)
                ax.contourf(xx, yy, Z_kde, levels=8, cmap=fade_cmap, alpha=0.7, zorder=2)
                ax.contour(xx, yy, Z_kde, levels=5, colors=_colors[ki],
                           linewidths=0.3, alpha=0.5, zorder=3)
            except Exception:
                pass
        ax.scatter(umap_xy[idx_k, 0], umap_xy[idx_k, 1], s=1.2,
                   color=_colors[ki], linewidths=0, alpha=0.85, zorder=4, rasterized=True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'C{ki}', fontsize=7, fontweight='bold', color=_colors[ki], pad=2)
        ax.axis('off')

    for _e in axes[n_show:]:
        _e.set_visible(False)

    fig.suptitle('Cluster density on UMAP', fontsize=8.5, fontweight='bold', y=0.96)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "umap_density.png")

    return fig


def plot_spatial_small_multiples(
    xy_plot: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    s_size: float,
    _colors: list,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 8: Per-cluster spatial distribution (small multiples)"""
    ncols_sm = min(n_clusters, 5)
    nrows_sm = int(np.ceil(n_clusters / ncols_sm))

    fig, axes = plt.subplots(nrows_sm, ncols_sm,
                              figsize=(2.0 * ncols_sm, 2.0 * nrows_sm),
                              gridspec_kw={'hspace': 0.08, 'wspace': 0.05,
                                           'left': 0.02, 'right': 0.98,
                                           'top': 0.90, 'bottom': 0.03})
    axes_flat = np.array(axes).flatten()

    for ki in range(n_clusters):
        ax = axes_flat[ki]
        idx_self = labels == ki
        ax.scatter(xy_plot[~idx_self, 0], xy_plot[~idx_self, 1],
                   s=s_size * 0.4, color='#e8e8e8', linewidths=0, rasterized=True, zorder=1)
        ax.scatter(xy_plot[idx_self, 0], xy_plot[idx_self, 1],
                   s=s_size * 0.9, color=_colors[ki], linewidths=0, rasterized=True, zorder=2)
        ax.axis('off')
        ax.set_title(f'C{ki} (n={idx_self.sum()})', fontsize=6, fontweight='bold',
                     color=_colors[ki], pad=2)

    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Spatial distribution per cluster', fontsize=8.5, fontweight='bold', y=0.96)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "spatial_small_multiples.png")

    return fig


def plot_refine(
    kref_df_plot: pd.DataFrame,
    N_CLUSTERS_SEL: int,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 9: K refinement decision curves"""
    fig, ax = plt.subplots(figsize=(3.5, 2.5),
                            gridspec_kw={'left': 0.14, 'right': 0.96, 'top': 0.87, 'bottom': 0.16})
    ks_ref = kref_df_plot["K"].to_numpy()

    metric_cfg_ref = [
        ("silhouette", "#E64B35", "o", "Silhouette"),
        ("spatial_edge_purity", "#4DBBD5", "s", "Spatial purity"),
        ("moran_median", "#00A087", "^", "Moran's I"),
        ("stability_ari", "#3C5488", "D", "Stability (ARI)")
    ]
    for col_name, color, marker, lab in metric_cfg_ref:
        if col_name in kref_df_plot.columns and kref_df_plot[col_name].notna().any():
            vals = kref_df_plot[col_name].to_numpy(dtype=float)
            vals_z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-9)
            ax.plot(ks_ref, vals_z, color=color, lw=1.0, marker=marker, ms=4,
                    markeredgecolor='white', markeredgewidth=0.3, label=lab, zorder=3)

    ax.axvline(int(N_CLUSTERS_SEL), ls=':', lw=1.0, color='#E64B35',
               label=f'Selected K={int(N_CLUSTERS_SEL)}')
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Z-score')
    ax.set_title('Top-K refinement', fontweight='bold', pad=5)
    ax.legend(loc='best', ncol=1, fontsize=5.5)
    despine(ax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "k_refine.png")

    return fig


def plot_summary_6panel(
    xy_plot: np.ndarray,
    labels: np.ndarray,
    umap_xy: np.ndarray,
    w_mod: np.ndarray,
    q: np.ndarray,
    n_clusters: int,
    sil_val: float,
    ari_val: float,
    nmi_val: float,
    edge_purity: float,
    s_size: float,
    _colors: list,
    log_df_plot: pd.DataFrame = None,
    save_dir: Path = None,
) -> plt.Figure:
    """Fig 10: Summary panel (6-panel)"""
    _c_arr = np.array([_colors[int(l)] for l in labels])

    fig = plt.figure(figsize=(7.2, 7.0))
    gs_main = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.35,
                                 left=0.08, right=0.96, top=0.94, bottom=0.06)

    # (a) Spatial clusters
    ax_a = fig.add_subplot(gs_main[0, 0])
    for k in range(n_clusters):
        idx = labels == k
        ax_a.scatter(xy_plot[idx, 0], xy_plot[idx, 1], s=s_size,
                     color=_colors[k], linewidths=0, rasterized=True)
    ax_a.axis('off')
    ax_a.set_title('Spatial domain map', fontweight='bold', pad=5)
    legend_patches = [mpatches.Patch(color=_colors[k], label=f'C{k}') for k in range(n_clusters)]
    ax_a.legend(handles=legend_patches, loc='lower left', fontsize=5, ncol=2,
                columnspacing=0.4, handlelength=0.8, handletextpad=0.2,
                bbox_to_anchor=(0.0, -0.02), frameon=False)

    # (b) UMAP
    ax_b = fig.add_subplot(gs_main[0, 1])
    order = np.random.permutation(len(labels))
    ax_b.scatter(umap_xy[order, 0], umap_xy[order, 1], s=s_size * 0.6,
                 c=np.array(_c_arr)[order], linewidths=0, alpha=0.8, rasterized=True)
    for k in range(n_clusters):
        idx = labels == k
        cx, cy = np.median(umap_xy[idx, 0]), np.median(umap_xy[idx, 1])
        ax_b.annotate(str(k), (cx, cy), fontsize=5.5, ha='center', va='center',
                      color='white', fontweight='bold',
                      bbox=dict(boxstyle='circle,pad=0.15', fc=_colors[k],
                                ec='white', lw=0.3, alpha=0.92))
    ax_b.set_xlabel('UMAP 1')
    ax_b.set_ylabel('UMAP 2')
    ax_b.set_title('UMAP embedding', fontweight='bold', pad=5)
    despine(ax_b)

    # (c) Modality weight
    ax_c = fig.add_subplot(gs_main[1, 0])
    _wlo = float(np.percentile(w_mod, 2))
    _whi = float(np.percentile(w_mod, 98))
    _wmd = float(np.clip(np.median(w_mod), _wlo + 1e-4, _whi - 1e-4))
    _norm_w = mcolors.TwoSlopeNorm(vmin=_wlo, vcenter=_wmd, vmax=_whi)
    sc_c = ax_c.scatter(xy_plot[:, 0], xy_plot[:, 1], s=s_size, c=w_mod,
                        cmap=PALETTE_DIVERGE, norm=_norm_w, linewidths=0, rasterized=True)
    ax_c.axis('off')
    ax_c.set_title(f'Modality weight (RNA) [{_wlo:.2f}--{_whi:.2f}]', fontweight='bold', pad=5)
    cb = plt.colorbar(sc_c, ax=ax_c, fraction=0.04, pad=0.02, shrink=0.78)
    cb.set_label(f'RNA weight (med={_wmd:.2f})', fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.3, length=2)
    cb.outline.set_linewidth(0.3)

    # (d) Training curves
    ax_d = fig.add_subplot(gs_main[1, 1])
    if log_df_plot is not None and "loss" in log_df_plot.columns:
        stage_color_map = {"pretrain": "#4DBBD5", "finetune": "#E64B35"}
        for st in log_df_plot["stage"].unique():
            sub = log_df_plot[log_df_plot["stage"] == st]
            ep = sub["epoch"].to_numpy() if "epoch" in sub.columns else sub.index.to_numpy()
            ax_d.plot(ep, sub["loss"].to_numpy(), color=stage_color_map.get(st, "#888888"),
                      lw=1.0, label=st)
        ax_d.set_xlabel('Epoch')
        ax_d.set_ylabel('Total loss')
        ax_d.set_title('Training curve', fontweight='bold', pad=5)
        ax_d.legend(loc='upper right', fontsize=5.5)
        despine(ax_d)
    else:
        ax_d.text(0.5, 0.5, 'No training log', ha='center', va='center',
                  transform=ax_d.transAxes, fontsize=7, color='#aaaaaa')
        ax_d.axis('off')

    # (e) Confidence boxplot
    ax_e = fig.add_subplot(gs_main[2, 0])
    conf_data = [q[labels == k].max(axis=1) for k in range(n_clusters)]
    bp = ax_e.boxplot(conf_data, patch_artist=True, notch=False, widths=0.6,
                      medianprops=dict(color='black', lw=0.8),
                      whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                      flierprops=dict(marker='.', ms=1.5, alpha=0.3, markeredgewidth=0))
    for patch, color in zip(bp['boxes'], _colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_linewidth(0.4)
    ax_e.set_xticks(range(1, n_clusters + 1))
    ax_e.set_xticklabels([f'C{k}' for k in range(n_clusters)], fontsize=5.5,
                          rotation=45 if n_clusters > 8 else 0)
    ax_e.set_xlabel('Cluster')
    ax_e.set_ylabel('Max soft-assign prob.')
    ax_e.set_title('Assignment confidence', fontweight='bold', pad=5)
    ax_e.set_ylim(0, 1.05)
    despine(ax_e)

    # (f) Cluster size pie chart
    ax_f = fig.add_subplot(gs_main[2, 1])
    cluster_sizes = [int((labels == k).sum()) for k in range(n_clusters)]
    wedges, texts, autotexts = ax_f.pie(
        cluster_sizes, labels=[f'C{k}' for k in range(n_clusters)],
        colors=_colors[:n_clusters], autopct='%1.1f%%',
        pctdistance=0.75, startangle=90, counterclock=False,
        wedgeprops=dict(linewidth=0.5, edgecolor='white')
    )
    for t in texts:
        t.set_fontsize(5.5)
    for t in autotexts:
        t.set_fontsize(4.5)
        t.set_color('white')
        t.set_fontweight('bold')
    ax_f.set_title('Cluster composition', fontweight='bold', pad=5)

    # Bottom metrics summary
    metrics_str = (f"N = {len(labels):,} spots | K = {n_clusters} clusters | "
                   f"Silhouette = {sil_val:.3f} | Spatial purity = {edge_purity:.3f}" +
                   (f" | ARI = {ari_val:.3f} | NMI = {nmi_val:.3f}" if not np.isnan(ari_val) else ""))
    fig.text(0.5, 0.005, metrics_str, ha='center', fontsize=6.0, color='#555555', style='italic')
    fig.suptitle('ProtoGlue Spatial Multi-omics Integration Summary',
                 fontsize=9.5, fontweight='bold', y=0.99)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_pub(save_dir / "summary_6panel.png")

    return fig


def plot_all_figures(
    xy_plot, labels, umap_xy, w_mod, q, n_clusters, sil_val, ari_val, nmi_val, edge_purity,
    s_size, _colors, log_df_plot, kscan_df_plot, kref_df_plot, adata_out, marker_genes, use_raw_marker,
    HEATMAP_CLIP, w_scale_rna, w_scale_om2, n_scales, save_dir
):
    """Helper: generate all figures at once (used by run_pipeline)."""
    plot_overview(xy_plot, labels, umap_xy, w_mod, n_clusters, sil_val, ari_val, nmi_val, edge_purity,
                  s_size, _colors, save_dir)
    plot_multiscale_attention(xy_plot, w_scale_rna, w_scale_om2, n_scales, s_size, save_dir)
    if kscan_df_plot is not None:
        plot_k_scan(kscan_df_plot, n_clusters, save_dir)
    if log_df_plot is not None:
        plot_train_loss(log_df_plot, save_dir)
    if marker_genes:
        plot_marker_heatmap(adata_out, "cluster_str", marker_genes, use_raw_marker, HEATMAP_CLIP, _colors, save_dir)
        plot_dotplot(adata_out, "cluster_str", marker_genes, use_raw_marker, HEATMAP_CLIP, save_dir)
    plot_umap_density(umap_xy, labels, n_clusters, _colors, save_dir)
    plot_spatial_small_multiples(xy_plot, labels, n_clusters, s_size, _colors, save_dir)
    if kref_df_plot is not None:
        plot_refine(kref_df_plot, n_clusters, save_dir)
    plot_summary_6panel(xy_plot, labels, umap_xy, w_mod, q, n_clusters, sil_val, ari_val, nmi_val,
                        edge_purity, s_size, _colors, log_df_plot, save_dir)


def export_metric_csv(
    adata_rna,
    labels,
    xy_plot,
    z,
    pca1,
    pca2,
    umap_xy=None,
    output_dir: Path = None,
    dataset_name: str = "dataset",
) -> Path:
    """Export unified evaluation table (spot_id, coords, labels, latent, PCA, UMAP, etc.)."""
    spot_id = pd.Index(adata_rna.obs_names.astype(str), name="spot_id")
    n_spots = len(spot_id)

    pred_label = pd.Series(labels.astype(int), index=spot_id, name="pred_label")

    # Handle true_label
    true_label = pd.Series([pd.NA] * n_spots, index=spot_id, name="true_label", dtype="object")
    if "manual_anno" in adata_rna.obs.columns:
        s = adata_rna.obs["manual_anno"]
        if s.notna().all():
            true_label = s.astype(str).rename("true_label")

    coords_df = pd.DataFrame(xy_plot, index=spot_id, columns=["x", "y"])
    latent_df = pd.DataFrame(z, index=spot_id, columns=[f"latent_{i+1}" for i in range(z.shape[1])])
    x1_df = pd.DataFrame(pca1, index=spot_id, columns=[f"x1_{i+1}" for i in range(pca1.shape[1])])
    x2_df = pd.DataFrame(pca2, index=spot_id, columns=[f"x2_{i+1}" for i in range(pca2.shape[1])])

    parts = [coords_df, pred_label.to_frame(), true_label.to_frame(), latent_df, x1_df, x2_df]
    if umap_xy is not None:
        umap_df = pd.DataFrame(umap_xy[:, :2], index=spot_id, columns=["umap1", "umap2"])
        parts.insert(3, umap_df)

    metric_df = pd.concat(parts, axis=1).reset_index()
    export_dir = output_dir / "metric_csv"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"{dataset_name}_metric_input.csv"
    metric_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path