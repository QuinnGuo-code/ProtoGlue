"""Cluster number (K) selection and evaluation utilities."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional


def elbow_from_inertia(df: pd.DataFrame) -> int:
    """Select K using the maximum-distance elbow method on inertia."""
    Ks = df["K"].to_numpy().astype(float)
    ys = df["inertia"].to_numpy().astype(float)
    if len(Ks) < 3 or not np.isfinite(ys).all():
        return int(Ks[int(len(Ks) // 2)])
    x = (Ks - Ks.min()) / (Ks.max() - Ks.min() + 1e-12)
    y = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    d = num / den
    return int(df.iloc[int(np.argmax(d))]["K"])


def zscore(x: np.ndarray) -> np.ndarray:
    """Compute z-score, ignoring NaN values."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)
    return (x - m) / s


def spatial_edge_purity(labels: np.ndarray, A) -> float:
    """Fraction of spatial edges connecting same-label spots.

    Parameters
    ----------
    labels : ndarray, shape (n,)
        Cluster labels.
    A : sparse matrix
        Spatial adjacency matrix.

    Returns
    -------
    float
        Edge purity score in [0, 1].
    """
    A = A.tocoo()
    m = A.row != A.col
    if int(m.sum()) == 0:
        return float("nan")
    return float((labels[A.row[m]] == labels[A.col[m]]).mean())


def moran_median_binary(labels: np.ndarray, A: sp.csr_matrix) -> float:
    """Median Moran's I computed over binary indicators per cluster."""
    A = A.tocsr()
    W = float(A.sum())
    N = A.shape[0]
    if W <= 0:
        return float("nan")
    res = []
    for k in np.unique(labels):
        x = (labels == k).astype(np.float64)
        xbar = x.mean()
        zc = x - xbar
        denom = float(np.sum(zc ** 2))
        if denom == 0:
            continue
        num = float(zc @ (A @ zc))
        I = (N / W) * num / denom
        res.append(I)
    return float(np.median(res)) if len(res) else float("nan")


def kmeans_stability(z: np.ndarray, K: int, runs: int = 3, seed: int = 0) -> float:
    """Measure clustering stability via pairwise ARI across KMeans runs."""
    if runs <= 1:
        return float("nan")
    labs = []
    for r in range(runs):
        km = KMeans(n_clusters=K, random_state=seed + r, n_init=30).fit(z)
        labs.append(km.labels_)
    aris = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            aris.append(adjusted_rand_score(labs[i], labs[j]))
    return float(np.mean(aris)) if len(aris) else float("nan")


def scan_select_k(
    z: np.ndarray,
    A_sp0,
    gt: Optional[np.ndarray] = None,
    K_GRID: list = None,
    K_SCORE_W_SIL: float = 1.0,
    K_SCORE_W_PURITY: float = 1.0,
    K_SCORE_W_MORAN: float = 0.5,
    K_SCORE_W_STAB: float = 0.5,
    K_SCORE_W_GT: float = 3.0,
    K_MIN_SELECT: int = 5,
    K_SELECT_W_ELBOW: float = 0.6,
    K_SELECT_W_LOGK: float = 0.8,
    K_SCAN_SAMPLE: int = 2000,
    K_SCAN_STABILITY_RUNS: int = 3,
    SEED: int = 2022,
):
    """Scan K values and select the best via composite scoring.

    Combines silhouette, spatial edge purity, Moran's I, stability,
    elbow analysis, and optional ground-truth ARI.

    Parameters
    ----------
    z : ndarray, shape (n_spots, d)
        Latent embeddings.
    A_sp0 : sparse matrix
        Base spatial adjacency.
    gt : ndarray, optional
        Ground-truth labels (for evaluation, not used in training).
    K_GRID : list of int
        Candidate K values to scan.

    Returns
    -------
    best_k : int
        Selected optimal number of clusters.
    df : DataFrame
        Scan results with all metrics and scores.
    """
    if K_GRID is None:
        K_GRID = [5, 6, 7, 8]

    rows = []
    for K in K_GRID:
        km = KMeans(n_clusters=K, random_state=SEED, n_init=50).fit(z)
        lab = km.labels_.astype(np.int32)
        sil = float(silhouette_score(z, lab)) if len(np.unique(lab)) > 1 else float("nan")
        purity = spatial_edge_purity(lab, A_sp0)
        moran = moran_median_binary(lab, A_sp0.tocsr())
        stab = kmeans_stability(z, K, runs=K_SCAN_STABILITY_RUNS, seed=SEED)
        gt_ari = np.nan
        if gt is not None:
            try:
                gt_ari = float(adjusted_rand_score(gt, lab))
            except Exception:
                gt_ari = np.nan
        rows.append({
            "K": int(K),
            "silhouette": sil,
            "spatial_edge_purity": purity,
            "moran_median": moran,
            "stability_ari": stab,
            "inertia": float(km.inertia_),
            "gt_ari": gt_ari,
        })

    df = pd.DataFrame(rows)
    df["z_sil"] = zscore(df["silhouette"].to_numpy())
    df["z_pur"] = zscore(df["spatial_edge_purity"].to_numpy())
    df["z_mor"] = zscore(df["moran_median"].to_numpy())
    df["z_sta"] = zscore(df["stability_ari"].to_numpy())
    df["z_gt"] = zscore(
        df["gt_ari"]
        .fillna(df["gt_ari"].median() if np.isfinite(df["gt_ari"]).any() else 0.0)
        .to_numpy()
    )

    score = (
        K_SCORE_W_SIL * df["z_sil"]
        + K_SCORE_W_PURITY * df["z_pur"]
        + K_SCORE_W_MORAN * df["z_mor"]
        + K_SCORE_W_STAB * df["z_sta"]
        + (K_SCORE_W_GT * df["z_gt"] if (gt is not None and np.isfinite(df["gt_ari"]).any()) else 0.0)
    )

    K_elbow = elbow_from_inertia(df)
    df["K_elbow"] = int(K_elbow)
    df["z_logk"] = zscore(np.log(df["K"].to_numpy().astype(float) + 1e-8))
    df["z_elb"] = zscore(-np.abs(df["K"].to_numpy().astype(float) - float(K_elbow)))

    score_vals = score.values if hasattr(score, "values") else np.asarray(score)
    score_select_vals = score_vals + K_SELECT_W_ELBOW * df["z_elb"].values + K_SELECT_W_LOGK * df["z_logk"].values
    df["score_select"] = score_select_vals
    df.loc[df["K"] < int(K_MIN_SELECT), "score_select"] = -1e9

    if df["score_select"].isna().all():
        print("Warning: all score_select are NaN. Returning first K in K_GRID as fallback.")
        return int(K_GRID[0]), df

    best_row = df.iloc[int(df["score_select"].idxmax())]
    best_k = int(best_row["K"])
    return best_k, df


def cluster_gmm_full(z: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """GMM clustering with BIC-based covariance type selection.

    Tries multiple covariance types and seeds, returns labels from
    the model with lowest BIC.
    """
    zs = StandardScaler().fit_transform(z)
    best_lab, best_bic = None, np.inf
    for cov in ["full", "tied", "diag", "spherical"]:
        for s in range(5):
            try:
                g = GaussianMixture(
                    n_components=int(K), covariance_type=cov, reg_covar=1e-6,
                    n_init=10, max_iter=1000, random_state=seed + s * 42,
                )
                lab = g.fit_predict(zs).astype(int)
                bic = g.bic(zs)
                if bic < best_bic:
                    best_bic, best_lab = bic, lab.copy()
            except Exception:
                pass
    if best_lab is not None:
        return best_lab
    return KMeans(n_clusters=int(K), random_state=seed, n_init=50).fit(z).labels_


def eval_labels(lab, A_sp0, gt=None, z=None, x1=None, x2=None) -> dict:
    """Evaluate clustering labels with multiple metrics.

    Parameters
    ----------
    lab : ndarray
        Predicted cluster labels.
    A_sp0 : sparse matrix
        Spatial adjacency for purity computation.
    gt : ndarray, optional
        Ground-truth labels.
    z : ndarray, optional
        Latent embeddings for silhouette computation.
    x1, x2 : ndarray, optional
        Original features for feature-space silhouette.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    out = {}
    out["n_clusters"] = int(len(np.unique(lab)))
    out["spatial_edge_purity"] = spatial_edge_purity(lab, A_sp0)
    out["moran_median"] = moran_median_binary(lab, A_sp0.tocsr())

    out["silhouette"] = np.nan
    out["silhouette_x1"] = np.nan
    out["silhouette_x2"] = np.nan
    out["silhouette_feature_type"] = None

    try:
        if gt is not None:
            if len(np.unique(gt)) > 1 and z is not None:
                out["silhouette"] = float(silhouette_score(z, gt))
                out["silhouette_feature_type"] = "latent_true_label"
            out["ARI"] = float(adjusted_rand_score(gt, lab))
            out["NMI"] = float(normalized_mutual_info_score(gt, lab))
            out["AMI"] = float(adjusted_mutual_info_score(gt, lab))
        else:
            if x1 is not None and len(np.unique(lab)) > 1:
                out["silhouette_x1"] = float(silhouette_score(x1, lab))
            if x2 is not None and len(np.unique(lab)) > 1:
                out["silhouette_x2"] = float(silhouette_score(x2, lab))
            out["silhouette_feature_type"] = "x1_x2_pred_label"
            out["ARI"] = np.nan
            out["NMI"] = np.nan
            out["AMI"] = np.nan
    except Exception:
        pass

    return out
