"""
Data Loaders & Validation Splitters
===================================

Provides standardized loaders for the extracted 132,740-node 3D CFD
benchmark dataset, as well as spatial and leave-one-geometry-out holdout
generators.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np


DEFAULT_FEATURE_NAMES = [
    "x_mm",
    "y_mm",
    "z_mm",
    "stenosis_fraction",
    "nominal_local_radius_mm",
    "relative_radial_position",
    "physics_axial_velocity_mm_s",
]

DEFAULT_CASE_NAMES = ["Healthy", "30%", "50%", "70%"]


def load_hemodyn_dataset(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads the golden-standard HemoDyn-AI ML dataset containing 132,740 points.

    Returns dictionary with:
      - 'X': (132740, 7) feature matrix
      - 'y': (132740,) CFD ground truth axial velocity (mm/s)
      - 'u_phys': (132740,) analytical Poiseuille axial velocity (mm/s)
      - 'residual': (132740,) physical discrepancy residual = y - u_phys
      - 'case_id': (132740,) integer index of case [0, 1, 2, 3]
      - 'feature_names': feature column names
      - 'case_names': list of case labels
    """
    if path is None:
        possible_paths = [
            Path(__file__).resolve().parent.parent.parent / "results" / "data" / "hemodyn_training_data.npz",
            Path("results/data/hemodyn_training_data.npz"),
            Path("results/ml/hemodyn_training_data.npz"),
        ]
        for p in possible_paths:
            if p.exists():
                path = p
                break
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                "Could not locate 'hemodyn_training_data.npz'. Please specify path."
            )

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y_axial"] if "y_axial" in data else data["y"]
    u_phys = data["physics_axial"] if "physics_axial" in data else data["u_phys"]
    case_id = data["case_id"]

    feature_names = (
        list(data["feature_names"]) if "feature_names" in data else
        list(data.get("features", DEFAULT_FEATURE_NAMES))
    )
    case_names = list(data["case_names"]) if "case_names" in data else DEFAULT_CASE_NAMES

    return {
        "X": X,
        "y": y,
        "u_phys": u_phys,
        "residual": y - u_phys,
        "case_id": case_id,
        "feature_names": feature_names,
        "case_names": case_names,
    }


def get_spatial_holdout_split(
    dataset: Dict[str, np.ndarray],
    holdout_z_min: float = 55.0,
    holdout_z_max: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions dataset by masking an entire axial section across all cases.
    Standard benchmark: z in [55, 75] mm (high-gradient narrowing and throat).
    """
    X = dataset["X"]
    y = dataset["y"]
    u_phys = dataset["u_phys"]
    z = X[:, 2]

    test_mask = (z >= holdout_z_min) & (z <= holdout_z_max)
    train_mask = ~test_mask

    return (
        X[train_mask],
        y[train_mask],
        u_phys[train_mask],
        X[test_mask],
        y[test_mask],
        u_phys[test_mask],
    )


def get_geometry_holdout_split(
    dataset: Dict[str, np.ndarray],
    held_out_case: str = "50%",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-One-Geometry-Out split: trains on 3 geometries and tests on the held-out geometry.
    Standard benchmark: held_out_case = '50%' (moderate stenosis).
    """
    X = dataset["X"]
    y = dataset["y"]
    u_phys = dataset["u_phys"]
    case_id = dataset["case_id"]

    case_names = dataset["case_names"]
    if held_out_case not in case_names:
        raise ValueError(f"Unknown case '{held_out_case}'. Available: {case_names}")

    case_idx = case_names.index(held_out_case)
    test_mask = case_id == case_idx
    train_mask = ~test_mask

    return (
        X[train_mask],
        y[train_mask],
        u_phys[train_mask],
        X[test_mask],
        y[test_mask],
        u_phys[test_mask],
    )
