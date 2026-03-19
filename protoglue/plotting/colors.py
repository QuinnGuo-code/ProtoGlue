"""Color palettes for ProtoGlue visualizations (colorblind-safe)."""

# Nature Reviews / nrc inspired palette
CLUSTER_COLORS = [
    "#E64B35",  # 0 vermillion
    "#4DBBD5",  # 1 cyan
    "#00A087",  # 2 teal
    "#3C5488",  # 3 navy
    "#F39B7F",  # 4 salmon
    "#8491B4",  # 5 slate blue
    "#91D1C2",  # 6 mint
    "#B09C85",  # 7 taupe
    "#7E6148",  # 8 brown
    "#E67E22",  # 9 tangerine
    "#9B59B6",  # 10 purple
    "#1ABC9C",  # 11 emerald
    "#DC0000",  # 12 scarlet
    "#2ECC71",  # 13 green
    "#F7DC6F",  # 14 gold
    "#34495E",  # 15 dark slate
]

PALETTE_DIVERGE = "RdBu_r"
PALETTE_SEQUENTIAL = "YlOrRd"
PALETTE_VIRIDIS = "viridis"
PALETTE_GREYS = "Greys"


def cluster_cmap(n):
    """Return a list of ``n`` colors, cycling through CLUSTER_COLORS.

    Parameters
    ----------
    n : int
        Number of clusters.

    Returns
    -------
    list of str
        Hex color strings.
    """
    return [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n)]
