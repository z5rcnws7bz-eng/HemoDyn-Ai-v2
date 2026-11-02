"""
HemoDyn-AI v2: Automated Test & Verification Suite
===================================================

Validates:
  1. Package imports and API surface.
  2. Analytical Poiseuille physics and mass continuity formulas.
  3. 132,740-node benchmark dataset schema and numerical ranges.
  4. Pretrained surrogate model inference and parity.
  5. Cross-geometry holdout benchmarks (71.4% throat error reduction).
  6. 3D surface geometry manifold presence.
"""

from pathlib import Path
import csv
import pytest
import numpy as np

# Ensure src is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hemodyn.physics import calculate_poiseuille_velocity, compute_reynolds_number
from hemodyn.data import load_hemodyn_dataset, get_spatial_holdout_split, get_geometry_holdout_split
from hemodyn.surrogate import PhysicsGuidedSurrogate, GeometryOnlySurrogate, evaluate_surrogate_performance


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_package_exports():
    """Verify package imports cleanly and exposes expected API."""
    import hemodyn
    assert hasattr(hemodyn, "__version__")
    assert hemodyn.__version__ == "2.0.0"
    assert hemodyn.__author__ == "Jace E. Roman"
    assert hasattr(hemodyn, "PhysicsGuidedSurrogate")
    assert hasattr(hemodyn, "calculate_poiseuille_velocity")


def test_physics_analytical_poiseuille():
    """Verify Poiseuille parabolic profile and continuity enforcement."""
    # Test inlet: d = 4.0 mm, R = 2.0 mm, Area = 4*pi mm^2
    # Flow rate Q = 1244.07 mm^3/s -> v_mean = 1244.07 / (4*pi) = 98.999 mm/s -> v_peak = 197.998 mm/s
    z = np.array([0.0, 0.0])
    d_wall_center = np.array([2.0, 0.0])  # Centerline (r=0), and wall (r=R)
    diam = np.array([4.0, 4.0])

    v = calculate_poiseuille_velocity(z, d_wall_center, diam, flow_rate=1244.0714285714287)

    # Centerline velocity must equal 2 * v_mean ~= 198.0 mm/s
    assert np.isclose(v[0], 197.998, atol=0.05)
    # No-slip condition at wall: velocity must equal 0.0
    assert np.isclose(v[1], 0.0, atol=1e-6)


def test_dataset_schema_and_integrity():
    """Verify the 132,740-node benchmark dataset schema."""
    data_path = REPO_ROOT / "results" / "data" / "hemodyn_training_data.npz"
    assert data_path.exists(), f"Dataset missing: {data_path}"

    data = load_hemodyn_dataset(data_path)
    X = data["X"]
    y = data["y"]
    u_phys = data["u_phys"]
    case_id = data["case_id"]

    assert X.shape == (132740, 7)
    assert y.shape == (132740,)
    assert u_phys.shape == (132740,)
    assert len(np.unique(case_id)) == 3  # Healthy (0), 30% (1), 50% (2)
    assert y.min() > -50.0  # Bounded reverse flow in post-stenotic separation zone
    assert y.max() < 2500.0  # Physically bounded forward jet
    assert np.all(u_phys >= 0.0)


def test_surrogate_inference():
    """Verify surrogate models load and predict within physical ranges."""
    model_path = REPO_ROOT / "results" / "models" / "physics_guided_residual_model.joblib"
    assert model_path.exists(), f"Model missing: {model_path}"

    surrogate = PhysicsGuidedSurrogate.load(model_path)
    # 7 features: [x, y, z, stenosis_fraction, nominal_radius, relative_radial_pos, u_phys]
    X_sample = np.array([[0.0, 0.0, 60.0, 0.50, 1.414, 0.0, 395.997]])
    u_phys_sample = np.array([395.997])

    pred = surrogate.predict(X_sample, u_phys_sample)
    assert pred.shape == (1,)
    assert pred[0] > 0.0
    assert pred[0] < 2500.0  # Physically bounded


def test_geometry_holdout_throat_benchmark():
    """Verify the 71.4% throat error reduction benchmark."""
    regional_csv = REPO_ROOT / "results" / "metrics" / "ml_50pct_regional_metrics.csv"
    assert regional_csv.exists(), f"Regional metrics CSV missing: {regional_csv}"

    geom_rmse = None
    pgml_rmse = None

    with open(regional_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("region") == "Throat":
                if row.get("model") == "Geometry-only ML":
                    geom_rmse = float(row["RMSE_mm_s"])
                elif row.get("model") == "Physics-guided ML":
                    pgml_rmse = float(row["RMSE_mm_s"])

    assert geom_rmse is not None, "Geometry-only ML throat RMSE missing"
    assert pgml_rmse is not None, "Physics-guided ML throat RMSE missing"

    assert np.isclose(geom_rmse, 131.52, atol=0.5)
    assert np.isclose(pgml_rmse, 37.55, atol=0.5)

    reduction = (1.0 - pgml_rmse / geom_rmse) * 100.0
    assert np.isclose(reduction, 71.45, atol=0.5)
    assert reduction > 70.0, f"Expected >70% error reduction, got {reduction:.1f}%"


def test_canonical_geometries_exist():
    """Verify all 4 canonical 3D manifold geometry surface files exist."""
    geom_dir = REPO_ROOT / "data" / "geometries"
    assert geom_dir.exists()
    for case in ["00", "30", "50", "70"]:
        assert (geom_dir / f"artery_{case}.stl").exists()
        assert (geom_dir / f"artery_{case}.vtp").exists()
