"""Plotting utility functions for ProtoGlue."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def make_colorbar(ax, cmap, vmin, vmax, label="", orientation="vertical"):
    """Add a colorbar to an axes."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation=orientation,
                      fraction=0.035, pad=0.03, shrink=0.82, aspect=20)
    cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=5, width=0.3, length=2)
    cb.outline.set_linewidth(0.3)
    return cb


def despine(ax, left=False, bottom=False):
    """Remove top/right spines (and optionally left/bottom)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def save_fig_pub(path, dpi=300, also_pdf=True, also_hires=True):
    """Save figure as PNG + PDF + high-resolution PNG (for PPT/poster)."""
    path = str(path)
    plt.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none", pad_inches=0.02)
    if also_pdf:
        pdf_path = path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.02)
    if also_hires:
        hires_path = path.replace(".png", "_hires.png")
        plt.savefig(hires_path, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.02)
    plt.close("all")


def add_scalebar(ax, xy, length_um=100, px_per_um=1.0, y_offset=0.03):
    """Add a spatial scale bar to an axes."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bar_len = length_um * px_per_um
    x0 = xmin + (xmax - xmin) * 0.05
    y0 = ymin + (ymax - ymin) * y_offset
    ax.plot([x0, x0 + bar_len], [y0, y0], color="black", lw=1.5, solid_capstyle="butt")
    ax.text(x0 + bar_len / 2, y0 - (ymax - ymin) * 0.02,
            f"{length_um} \u00b5m", ha="center", va="top", fontsize=5, color="black")


def _to_1d(arr, n, name="arr"):
    """Flatten array to 1D vector of length n (handles (n,1) or (n,2) inputs)."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[0] != n:
            a = a.reshape(n, -1)
        if a.shape[1] == 1:
            return a[:, 0]
        # If two columns, take the first (RNA side by convention)
        return a[:, 0]
    return a.reshape(n)


def save_fig(path: Path, dpi: int = 220):
    """Simple figure save (legacy compatibility)."""
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def fig_name(fig_dir: Path, section: str, plot: str, sample: str = "ALL",
             method: str = None, part: int = 0):
    """Generate a standardized figure filename."""
    s = f"fig__{section}__{plot}__sample-{sample}__part{part}"
    if method:
        s += f"__method-{method}"
    return fig_dir / f"{s}.png"
