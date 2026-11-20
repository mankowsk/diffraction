"""Reciprocal lattice points with structure-factor intensities and the Ewald condition."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal
    from diffraction.core.lab_geometry import LabGeometry


def reciprocal_lattice_points(
    crystal: "Crystal",
    energy: float,
    q_max: float | None = None,
    min_intensity: float = 1e-3,
    max_points: int | None = 3000,
    occupancy: bool = True,
    debye_waller: bool = True,
) -> pd.DataFrame:
    """Reflections with |Q| <= q_max and their intensities |F|^2.

    Args:
        crystal: Crystal with a UB matrix; structure factors need a CIF (``from_cif``),
            otherwise all intensities are 1
        energy: X-ray energy in eV (form factors; default ``q_max``)
        q_max: Maximum |Q| in 1/Angstrom (2 pi convention); default 2 k = 4 pi / lambda,
            i.e. all reflections reachable at this energy
        min_intensity: Drop reflections with |F|^2 below this fraction of the strongest
            (removes systematic absences)
        max_points: Keep at most this many reflections (the strongest); None for all
        occupancy, debye_waller: Passed to ``Crystal.calc_structure_factors``

    Returns:
        DataFrame: columns ``h, k, l`` (int), ``q`` (|Q|, 1/Angstrom), ``F`` (complex),
        ``intensity`` (|F|^2), ``rel_intensity`` (|F|^2 / max), sorted by ``q``; (000) excluded
    """
    UB = np.asarray(crystal.ubcalc.UB, dtype=float)
    if q_max is None:
        q_max = 2 * 2 * np.pi / crystal.en2lam(energy)

    # |h| = |G . a| / 2 pi <= q_max |a| / 2 pi
    lattice = 2 * np.pi * np.linalg.inv(UB).T
    limits = np.floor(q_max * np.linalg.norm(lattice, axis=0) / (2 * np.pi)).astype(int)
    grids = np.meshgrid(*(np.arange(-n, n + 1) for n in limits), indexing="ij")
    hkls = np.stack([g.ravel() for g in grids], axis=1)
    q = np.linalg.norm(hkls @ UB.T, axis=1)
    keep = (q <= q_max) & np.any(hkls != 0, axis=1)
    hkls, q = hkls[keep], q[keep]

    if crystal.xucalc is not None:
        F = crystal.calc_structure_factors(hkls, energy, occupancy=occupancy, debye_waller=debye_waller)
    else:
        F = np.ones(len(hkls), dtype=complex)
    intensity = np.abs(F) ** 2
    rel = intensity / intensity.max() if len(intensity) and intensity.max() > 0 else intensity

    df = pd.DataFrame({"h": hkls[:, 0], "k": hkls[:, 1], "l": hkls[:, 2], "q": q, "F": F,
                       "intensity": intensity, "rel_intensity": rel})
    df = df[df["rel_intensity"] >= min_intensity]
    if max_points is not None and len(df) > max_points:
        df = df.nlargest(max_points, "intensity")
    return df.sort_values("q").reset_index(drop=True)


def reflections_in_condition(geometry: "LabGeometry", points: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Reflections within ``tolerance`` of the Ewald sphere, closest first.

    Args:
        geometry: Lab geometry at the current angles (``Crystal.calc_lab_geometry``)
        points: Reflections as from :func:`reciprocal_lattice_points` (columns ``h, k, l``, ...)
        tolerance: Maximum |excitation error| in 1/Angstrom

    Returns:
        DataFrame: ``points`` rows plus ``excitation_error`` (1/Angstrom), the detector
        angles ``delta``, ``gamma``, ``two_theta`` (degrees) at which each reflection appears
        and ``kx, ky, kz``: unit vector of the diffracted beam k_i + G (diffcalc frame)
    """
    hkls = points[["h", "k", "l"]].to_numpy(dtype=float)
    df = points.assign(excitation_error=geometry.excitation_error(hkls))
    df = df[df["excitation_error"].abs() <= tolerance]
    df = df.reindex(df["excitation_error"].abs().sort_values().index)
    hkls = df[["h", "k", "l"]].to_numpy(dtype=float)
    delta, gamma, two_theta = geometry.detector_angles(hkls)
    k_f = hkls @ geometry.reciprocal_lattice.T + geometry.k_i
    u = k_f / np.linalg.norm(k_f, axis=1, keepdims=True) if len(df) else np.zeros((0, 3))
    return df.assign(delta=delta, gamma=gamma, two_theta=two_theta, kx=u[:, 0], ky=u[:, 1], kz=u[:, 2])
