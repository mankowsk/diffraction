"""Scattering geometry in the laboratory frame for a set of diffractometer angles.

All quantities are computed in the diffcalc frame (see :mod:`diffraction.core.frames`) with
diffcalc's own rotation matrices, so they are consistent with ``Crystal.calc_angles`` and
``Crystal.calc_hkl`` by construction. Use :meth:`LabGeometry.in_frame` to read them in
another frame.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from diffcalc.hkl.geometry import Position, get_rotation_matrices

from diffraction.core.frames import convert

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

ANGLE_NAMES = ("mu", "delta", "gamma", "eta", "chi", "phi")


def to_position(angles: Position | Mapping[str, Any]) -> Position:
    """Diffcalc ``Position`` (degrees) from a Position, dict or ``calc_angles`` DataFrame row.

    ``gamma`` and ``nu`` are accepted as names for the same angle; missing angles are zero.
    """
    if isinstance(angles, Position):
        return angles
    values = {k: float(angles[k]) for k in angles.keys() if k in ANGLE_NAMES + ("nu",)}
    if "gamma" in values:
        values["nu"] = values.pop("gamma")
    return Position(**{k: values.get(k, 0.0) for k in ("mu", "delta", "nu", "eta", "chi", "phi")})


@dataclass
class LabGeometry:
    """Scattering geometry for one diffractometer position, in the diffcalc frame.

    Attributes:
        position: Diffractometer angles (degrees)
        energy: X-ray energy in eV
        wavelength: X-ray wavelength in Angstrom
        R_sample: Sample rotation MU @ ETA @ CHI @ PHI (phi frame -> lab)
        R_detector: Detector rotation NU @ DELTA (lab)
        k_i: Incident wave vector (1/Angstrom, |k| = 2 pi / lambda)
        k_f: Scattered wave vector (1/Angstrom)
        Q: Scattering vector k_f - k_i (1/Angstrom)
        two_theta: Scattering angle in degrees
        lattice: Real-space lattice vectors a, b, c as columns, rotated into the lab (Angstrom)
        reciprocal_lattice: Reciprocal lattice vectors a*, b*, c* as columns in the lab
            (1/Angstrom, including 2 pi): R_sample @ UB, so Q = reciprocal_lattice @ hkl
        surface_normal: Unit surface normal of the sample in the lab
        hkl: Miller indices at this position, UB^-1 Q (non-integer in general)
    """

    position: Position
    energy: float
    wavelength: float
    R_sample: np.ndarray
    R_detector: np.ndarray
    k_i: np.ndarray
    k_f: np.ndarray
    Q: np.ndarray
    two_theta: float
    lattice: np.ndarray
    reciprocal_lattice: np.ndarray
    surface_normal: np.ndarray
    hkl: np.ndarray

    def in_frame(self, frame: str = "swissfel") -> dict[str, np.ndarray]:
        """Lab vectors and rotations converted to ``frame``.

        Returns:
            dict: ``k_i``, ``k_f``, ``Q``, ``surface_normal`` (vectors), ``lattice``
            (columns a, b, c), ``reciprocal_lattice`` (columns a*, b*, c*), ``R_sample`` and
            ``R_detector`` (rotation matrices)
        """
        out = {k: convert(getattr(self, k), "diffcalc", frame) for k in ("k_i", "k_f", "Q", "surface_normal")}
        for k in ("lattice", "reciprocal_lattice"):
            out[k] = convert(getattr(self, k).T, "diffcalc", frame).T
        for k in ("R_sample", "R_detector"):
            out[k] = convert(getattr(self, k), "diffcalc", frame, kind="rotation")
        return out


def calc_lab_geometry(crystal: "Crystal", angles: Position | Mapping[str, Any], energy: float) -> LabGeometry:
    """Compute the lab-frame scattering geometry of ``crystal`` at ``angles``.

    Args:
        crystal: Crystal with a UB matrix (``calc_ub``)
        angles: diffcalc ``Position``, dict of angles (degrees; ``gamma`` or ``nu``) or one
            row of ``crystal.calc_angles(...)``
        energy: X-ray energy in eV

    Returns:
        LabGeometry: Geometry in the diffcalc frame
    """
    UB = crystal.ubcalc.UB
    if UB is None:
        raise ValueError("Crystal has no UB matrix; call calc_ub() first.")
    UB = np.asarray(UB, dtype=float)

    position = to_position(angles)
    MU, DELTA, NU, ETA, CHI, PHI = get_rotation_matrices(Position.asradians(position))
    R_sample = MU @ ETA @ CHI @ PHI
    R_detector = NU @ DELTA

    wavelength = crystal.en2lam(energy)
    k = 2 * np.pi / wavelength
    k_i = np.array([0.0, k, 0.0])  # incident beam along diffcalc +y
    k_f = R_detector @ k_i
    Q = k_f - k_i

    # Columns a, b, c in the phi frame satisfy A^T @ UB = 2 pi I
    lattice = R_sample @ (2 * np.pi * np.linalg.inv(UB).T)
    surface_normal = R_sample @ np.asarray(crystal.ubcalc.surf_nphi, dtype=float).ravel()
    surface_normal /= np.linalg.norm(surface_normal)
    hkl = np.linalg.solve(UB, R_sample.T @ Q)

    return LabGeometry(
        position=position,
        energy=energy,
        wavelength=wavelength,
        R_sample=R_sample,
        R_detector=R_detector,
        k_i=k_i,
        k_f=k_f,
        Q=Q,
        two_theta=float(np.degrees(np.arccos(np.clip(k_f @ k_i / k**2, -1.0, 1.0)))),
        lattice=lattice,
        reciprocal_lattice=R_sample @ UB,
        surface_normal=surface_normal,
        hkl=hkl,
    )
