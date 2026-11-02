"""
Publication-Grade Hemodynamic Plotting Utilities
================================================

Configures matplotlib styling and provides clean plotting methods.
"""

from typing import Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def set_publication_style():
    """Sets consistent publication-grade styling for scientific figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_cross_geometry_comparison(
    metrics_csv_path: Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Renders bar comparison of Physics baseline, Geometry ML, and PGML across holdout geometries.
    """
    import pandas as pd
    set_publication_style()

    df = pd.read_csv(metrics_csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["physics_rmse"], width, label="Analytical Physics Prior", color="#4a7c59", alpha=0.85)
    ax.bar(x, df["geom_ml_rmse"], width, label="Geometry-Only ML (Black-box)", color="#d95f02", alpha=0.85)
    ax.bar(x + width, df["pgml_rmse"], width, label="Physics-Guided ML (Ours)", color="#1b7837", alpha=0.95)

    ax.set_ylabel("Holdout RMSE (mm/s)")
    ax.set_title("Leave-One-Geometry-Out Surrogate Generalization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["geometry"])
    ax.legend(frameon=True)

    if output_path:
        fig.savefig(output_path)
    return fig
