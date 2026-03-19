import matplotlib as mpl

# Global typography settings (Nature style)
mpl.rcParams.update({
    # ── Font ──────────────────────────────────────
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",

    # ── Font size (Nature: title 8pt, label 7pt, tick 6pt) ──
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6.5,

    # ── Line widths ─────────────────────────────────────
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,

    # ── Ticks ─────────────────────────────────────
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,

    # ── Legend ─────────────────────────────────────
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    "legend.borderaxespad": 0.3,

    # ── Saving ─────────────────────────────────────
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed fonts (journal requirement)
    "ps.fonttype": 42,

    # ── Background / spines ────────────────────────────────
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})