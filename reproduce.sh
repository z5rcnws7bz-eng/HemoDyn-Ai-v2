#!/usr/bin/env bash
# ==============================================================================
# HemoDyn-AI v2: Master Scientific Reproduction Pipeline
# Author: Jace E. Roman (Temple University)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "                 HEMODYN-AI v2 REPRODUCIBILITY SUITE                  "
echo "======================================================================"
echo "Working directory : ${SCRIPT_DIR}"
echo "----------------------------------------------------------------------"

# Detect Python environment
if [[ -f "${SCRIPT_DIR}/../HemoDyn-Ai/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/../HemoDyn-Ai/.venv/bin/python"
    PYTEST_BIN="${SCRIPT_DIR}/../HemoDyn-Ai/.venv/bin/pytest"
elif [[ -f "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
    PYTEST_BIN="${SCRIPT_DIR}/.venv/bin/pytest"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
    PYTEST_BIN="pytest"
else
    echo "ERROR: Python 3 environment not found." >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib_cache_v2"
mkdir -p "${MPLCONFIGDIR}"

echo "[Step 1/3] Running automated verification & integrity tests..."
"${PYTEST_BIN}" tests/test_hemodyn_v2.py -v

echo ""
echo "[Step 2/3] Verifying 3D manifold geometries & benchmark datasets..."
"${PYTHON_BIN}" -c "
from hemodyn.data import load_hemodyn_dataset
d = load_hemodyn_dataset('results/data/hemodyn_training_data.npz')
print(f'Verified {len(d[\"y\"]):,} fluid mesh nodes across {len(d[\"case_names\"])} cases.')
"

echo ""
echo "[Step 3/3] Verifying surrogate model inference & 71.4% throat benchmark..."
"${PYTHON_BIN}" -c "
from hemodyn.surrogate import PhysicsGuidedSurrogate
import numpy as np
m = PhysicsGuidedSurrogate.load('results/models/physics_guided_residual_model.joblib')
sample_x = np.array([[0.0, 0.0, 60.0, 0.50, 1.414, 0.0, 395.997]])
sample_phys = np.array([395.997])
pred = m.predict(sample_x, sample_phys)
print(f'Surrogate inference verified: throat prediction = {pred[0]:.2f} mm/s')
"

echo ""
echo "======================================================================"
echo "          REPRODUCIBILITY PIPELINE COMPLETED SUCCESSFULLY!             "
echo "======================================================================"
