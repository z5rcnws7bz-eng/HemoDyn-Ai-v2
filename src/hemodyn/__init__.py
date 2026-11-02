"""
HemoDyn-AI: Physics-Guided Machine Learning (PGML) for 3D Hemodynamics
=====================================================================

A scientific benchmark package coupling Navier-Stokes finite element
computational fluid dynamics (CFD) with physics-guided machine learning
surrogates for blood flow across arterial stenoses.

Author: Jace E. Roman (Temple University)
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Jace E. Roman"
__email__ = "jace.roman@temple.edu"

from .physics import (
    calculate_poiseuille_velocity,
    compute_reynolds_number,
    compute_womersley_number,
)
from .data import (
    load_hemodyn_dataset,
    get_spatial_holdout_split,
    get_geometry_holdout_split,
)
from .surrogate import (
    PhysicsGuidedSurrogate,
    GeometryOnlySurrogate,
    evaluate_surrogate_performance,
)

__all__ = [
    "calculate_poiseuille_velocity",
    "compute_reynolds_number",
    "compute_womersley_number",
    "load_hemodyn_dataset",
    "get_spatial_holdout_split",
    "get_geometry_holdout_split",
    "PhysicsGuidedSurrogate",
    "GeometryOnlySurrogate",
    "evaluate_surrogate_performance",
]

