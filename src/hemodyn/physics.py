"""
Analytical Physics & Fluid Mechanics Formulations
=================================================

Provides ground-truth analytical baselines including:
  1. Laminar Poiseuille parabolic profile with continuity-enforced flow rate Q.
  2. Dimensionless hemodynamic numbers (Reynolds, Womersley).
"""

from typing import Union
import numpy as np


# Standard default hemodynamic constants
DEFAULT_FLOW_RATE = 1244.0714285714287  # mm^3/s (equivalent to 1.244 ml/s)
DEFAULT_BLOOD_DENSITY = 1.06e-3          # g/mm^3 (equivalent to 1060 kg/m^3)
DEFAULT_DYNAMIC_VISCOSITY = 0.004        # Pa*s (0.004 N*s/m^2 = 4 cP)


def calculate_poiseuille_velocity(
    z: Union[float, np.ndarray],
    d_wall: Union[float, np.ndarray],
    local_diam: Union[float, np.ndarray],
    flow_rate: float = DEFAULT_FLOW_RATE,
) -> np.ndarray:
    """
    Computes local analytical Poiseuille parabolic velocity based on local
    vessel radius and distance from the vessel wall, strictly enforcing
    mass flow continuity Q:

        R(z) = local_diam / 2
        r(z, d_wall) = max(0, R(z) - d_wall)
        v_mean(z) = Q / (pi * R(z)^2)
        v_peak(z) = 2 * v_mean(z)
        v(r, z) = v_peak(z) * [1 - (r(z) / R(z))^2]

    Parameters
    ----------
    z : float or np.ndarray
        Longitudinal axial coordinate (mm).
    d_wall : float or np.ndarray
        Euclidean distance to the nearest arterial wall boundary (mm).
    local_diam : float or np.ndarray
        Local arterial lumen diameter at position z (mm).
    flow_rate : float, optional
        Prescribed volumetric flow rate in mm^3/s (default: 1244.07 mm^3/s).

    Returns
    -------
    np.ndarray
        Analytical axial velocity profile in mm/s (bounded to >= 0).
    """
    z = np.asarray(z, dtype=np.float64)
    d_wall = np.asarray(d_wall, dtype=np.float64)
    local_diam = np.asarray(local_diam, dtype=np.float64)

    radius = local_diam / 2.0
    # Guard against non-positive radii
    radius = np.maximum(radius, 1e-6)

    # Radial distance from the vessel centerline: r = R - d_wall
    r = np.maximum(0.0, radius - d_wall)
    r_norm = np.clip(r / radius, 0.0, 1.0)

    # Mean and centerline peak velocity enforcing continuity Q = v_mean * Area
    area = np.pi * (radius ** 2)
    v_mean = flow_rate / area
    v_peak = 2.0 * v_mean

    # Parabolic Poiseuille distribution
    v_poiseuille = v_peak * (1.0 - (r_norm ** 2))

    return np.maximum(0.0, v_poiseuille)


def compute_reynolds_number(
    velocity: Union[float, np.ndarray],
    diameter: Union[float, np.ndarray],
    rho: float = DEFAULT_BLOOD_DENSITY,
    mu: float = DEFAULT_DYNAMIC_VISCOSITY,
) -> np.ndarray:
    """
    Computes local Reynolds number Re = (rho * v * D) / mu.

    Parameters
    ----------
    velocity : float or np.ndarray
        Mean or peak velocity (mm/s).
    diameter : float or np.ndarray
        Lumen diameter (mm).
    rho : float
        Fluid density (g/mm^3).
    mu : float
        Dynamic viscosity (g / (mm * s)).

    Returns
    -------
    np.ndarray
        Dimensionless Reynolds number.
    """
    v = np.asarray(velocity, dtype=np.float64)
    d = np.asarray(diameter, dtype=np.float64)
    # Convert mu from Pa*s (kg/(m*s)) to g/(mm*s): 1 Pa*s = 1e-3 g/(mm*s)
    mu_local = mu * 1e-3
    return (rho * v * d) / mu_local


def compute_womersley_number(
    radius: float,
    heart_rate_bpm: float = 75.0,
    rho: float = DEFAULT_BLOOD_DENSITY,
    mu: float = DEFAULT_DYNAMIC_VISCOSITY,
) -> float:
    """
    Computes dimensionless Womersley number alpha = R * sqrt(omega * rho / mu).
    """
    omega = 2.0 * np.pi * (heart_rate_bpm / 60.0)
    mu_local = mu * 1e-3
    return float(radius * np.sqrt(omega * rho / mu_local))

