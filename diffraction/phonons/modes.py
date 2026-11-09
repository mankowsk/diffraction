"""Phonon mode representation and convention handling.

All phonon modes are stored in one canonical form:

- **Mass-normalized displacement** ``u``: the physical displacement pattern of
  the atoms, normalized such that ``sum_n m_n |u_n|^2 = 1`` (Cartesian norm,
  masses in amu). A physical displacement is ``Q * u`` with the normal
  coordinate ``Q`` in Angstrom * sqrt(amu).
- **Reduced coordinates**: ``u`` is stored in units of the lattice vectors, so it
  is independent of the Cartesian frame (e.g. DFT frame vs. xrayutilities frame)
  and can be added directly to reduced atomic positions.

The DFT-style eigenvector ``e = sqrt(m) * u_cart`` (normalized as
``sum_n |e_n|^2 = 1``) is available as a derived property. At the Gamma point,
``u`` conserves the center of mass (COM) for optical modes, whereas raw
eigenvectors ``e`` do not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: Atomic mass unit in kg
AMU_KG = 1.66053906660e-27

#: Tolerance on the normalized COM residual for a mode to count as COM-conserving
COM_TOL = 1e-3

#: Relative tolerance for detecting a rigid translation (acoustic mode at Gamma)
ACOUSTIC_TOL = 1e-2

_COORDS = ("reduced", "cartesian")
_KINDS = ("displacement", "eigenvector")
_CHECKS = ("warn", "raise", "off")


class PhononConsistencyWarning(UserWarning):
    """Warning emitted when a phonon mode fails a consistency check."""


@dataclass(eq=False)
class PhononMode:
    """A phonon mode stored as mass-normalized displacement in reduced coordinates.

    Use :meth:`from_vectors` to construct a mode from eigenvectors or
    displacements in reduced or Cartesian coordinates. The direct constructor
    expects data that is already in canonical form.

    Attributes:
        name: Identifier for the phonon mode
        displacement: Complex array (n_atoms x 3), reduced coordinates,
            normalized as ``sum_n m_n |u_n|^2 = 1`` (Cartesian norm)
        masses: Atomic masses in amu (n_atoms)
        lattice: Real-space lattice (3 x 3, rows a1, a2, a3) in Angstrom that
            defines the reduced coordinates
        atom_names: Optional element symbols (n_atoms)
        frequency: Mode frequency in THz (optional)
        q: Wave vector in reduced reciprocal coordinates
        source: Free-text origin of the mode (e.g. ``"phonopy:band.yaml q0 b5"``)
        import_report: Details recorded during import (atom mapping, origin
            shift, lattice mismatch, ...)
        consistency: Result of :meth:`check`, recomputed on construction
    """

    name: str
    displacement: np.ndarray
    masses: np.ndarray
    lattice: np.ndarray
    atom_names: list[str] | None = None
    frequency: float | None = None
    q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    source: str | None = None
    import_report: dict | None = None
    consistency: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.displacement = np.asarray(self.displacement, dtype=complex)
        self.masses = np.asarray(self.masses, dtype=float)
        self.lattice = np.asarray(self.lattice, dtype=float)
        self.q = np.asarray(self.q, dtype=float)

        n_atoms = len(self.masses)
        if self.displacement.shape != (n_atoms, 3):
            raise ValueError(
                f"displacement must have shape ({n_atoms}, 3), got {self.displacement.shape}"
            )
        if self.lattice.shape != (3, 3):
            raise ValueError(f"lattice must have shape (3, 3), got {self.lattice.shape}")
        if self.atom_names is not None and len(self.atom_names) != n_atoms:
            raise ValueError(
                f"atom_names has {len(self.atom_names)} entries, expected {n_atoms}"
            )

        self.consistency = self.check()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_vectors(
        cls,
        name: str,
        vectors: np.ndarray | list,
        masses: np.ndarray | list[float],
        lattice: np.ndarray,
        *,
        coords: str = "reduced",
        kind: str = "displacement",
        frame_lattice: np.ndarray | None = None,
        q: np.ndarray | list[float] = (0.0, 0.0, 0.0),
        frequency: float | None = None,
        atom_names: list[str] | None = None,
        source: str | None = None,
        import_report: dict | None = None,
        check: str = "warn",
    ) -> "PhononMode":
        """Create a phonon mode from vectors in any supported convention.

        The vectors are converted to the canonical form (mass-normalized
        displacement in reduced coordinates of ``lattice``). Their overall
        scale is irrelevant; the global phase is fixed such that the largest
        component is real and positive.

        Args:
            name: Identifier for the phonon mode
            vectors: Array (n_atoms x 3), real or complex
            masses: Atomic masses in amu (n_atoms)
            lattice: Real-space lattice (rows a1, a2, a3) in Angstrom of the crystal
                the mode belongs to
            coords: ``"reduced"`` or ``"cartesian"``
            kind: ``"displacement"`` (physical displacement pattern, any
                normalization) or ``"eigenvector"`` (mass-weighted dynamical
                matrix eigenvector as produced by DFT codes, ``e = sqrt(m) u``)
            frame_lattice: Lattice defining the frame of the input vectors (rows
                a1, a2, a3). For Cartesian input from a DFT calculation, pass the
                DFT lattice. Defaults to ``lattice``.
            q: Wave vector in reduced reciprocal coordinates
            frequency: Mode frequency in THz
            atom_names: Optional element symbols
            source: Free-text origin of the mode
            import_report: Details recorded during import
            check: What to do if a Gamma-point optical mode does not conserve the
                center of mass: ``"warn"``, ``"raise"`` or ``"off"``

        Returns:
            PhononMode in canonical form
        """
        if coords not in _COORDS:
            raise ValueError(f"coords must be one of {_COORDS}, got '{coords}'")
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got '{kind}'")
        if check not in _CHECKS:
            raise ValueError(f"check must be one of {_CHECKS}, got '{check}'")

        if frame_lattice is not None and coords == "reduced":
            raise ValueError("frame_lattice is only meaningful for coords='cartesian'")

        vectors = np.asarray(vectors, dtype=complex)
        masses = np.asarray(masses, dtype=float)
        lattice = np.asarray(lattice, dtype=float)
        frame_lattice = lattice if frame_lattice is None else np.asarray(frame_lattice, dtype=float)

        if vectors.shape != (len(masses), 3):
            raise ValueError(
                f"vectors must have shape ({len(masses)}, 3), got {vectors.shape}"
            )

        # Bring the vectors to reduced coordinates. The reduced coordinates are
        # frame independent; only the lattice used for the conversion matters.
        if coords == "cartesian":
            vectors = vectors @ np.linalg.inv(frame_lattice)

        # Eigenvector -> displacement: u = e / sqrt(m). This is a per-atom scalar,
        # so it can be applied in reduced coordinates as well.
        if kind == "eigenvector":
            vectors = vectors / np.sqrt(masses)[:, np.newaxis]

        displacement = _canonicalize(vectors, masses, lattice)

        mode = cls(
            name=name,
            displacement=displacement,
            masses=masses,
            lattice=lattice,
            atom_names=atom_names,
            frequency=frequency,
            q=np.asarray(q, dtype=float),
            source=source,
            import_report=import_report,
        )

        if kind == "displacement" and mode.consistency.get("conserves_com") is False:
            # Hint: the input may have been eigenvectors passed as displacements
            alt = _canonicalize(vectors / np.sqrt(masses)[:, np.newaxis], masses, lattice)
            alt_residual = _com_residual(alt @ lattice, masses)
            if alt_residual < COM_TOL:
                mode.consistency["hint"] = (
                    "The vectors conserve the COM when interpreted as eigenvectors; "
                    "use kind='eigenvector'."
                )

        if check != "off" and mode.consistency.get("conserves_com") is False:
            msg = (
                f"Phonon mode '{name}' does not conserve the center of mass "
                f"(normalized COM residual {mode.consistency['com_residual']:.2e})."
            )
            if "hint" in mode.consistency:
                msg += " " + mode.consistency["hint"]
            if check == "raise":
                raise ValueError(msg)
            warnings.warn(msg, PhononConsistencyWarning, stacklevel=2)

        return mode

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the mode."""
        return len(self.masses)

    @property
    def is_gamma(self) -> bool:
        """Whether the mode is at the Gamma point (q = 0 mod reciprocal lattice)."""
        return bool(np.allclose(self.q - np.round(self.q), 0.0, atol=1e-8))

    @property
    def displacement_reduced(self) -> np.ndarray:
        """Mass-normalized displacement in reduced coordinates (the stored array)."""
        return self.displacement

    @property
    def displacement_cartesian(self) -> np.ndarray:
        """Mass-normalized displacement in Cartesian coordinates of ``lattice``.

        Units are 1/sqrt(amu); multiply by the normal coordinate Q (Angstrom *
        sqrt(amu)) to obtain displacements in Angstrom.
        """
        return self.displacement @ self.lattice

    @property
    def eigenvector(self) -> np.ndarray:
        """DFT-style eigenvector ``e = sqrt(m) u`` in Cartesian coordinates, ``sum |e|^2 = 1``."""
        return np.sqrt(self.masses)[:, np.newaxis] * self.displacement_cartesian

    def real_displacement_reduced(self) -> np.ndarray:
        """Real mass-normalized displacement in reduced coordinates for Gamma-point modes.

        Raises:
            ValueError: If the mode is not at the Gamma point
        """
        if not self.is_gamma:
            raise ValueError(
                f"Phonon mode '{self.name}' has q = {self.q.tolist()}; only Gamma-point "
                "modes are supported for structure factors and visualization."
            )
        return self.displacement.real

    def real_displacement_cartesian(self) -> np.ndarray:
        """Real mass-normalized displacement in Cartesian coordinates for Gamma-point modes."""
        return self.real_displacement_reduced() @ self.lattice

    def amplitude_for_max_displacement(self, distance: float) -> float:
        """Normal coordinate Q at which the largest atomic displacement equals ``distance``.

        Args:
            distance: Maximum atomic displacement in Angstrom

        Returns:
            float: Normal coordinate Q in Angstrom * sqrt(amu)
        """
        max_disp = np.max(np.linalg.norm(np.abs(self.displacement_cartesian), axis=1))
        return float(distance / max_disp)

    def check(self) -> dict[str, object]:
        """Evaluate the consistency of the mode.

        Returns:
            dict with keys:
                ``is_gamma``: mode is at the Gamma point
                ``norm``: ``sum_n m_n |u_n|^2`` (should be 1)
                ``com_residual``: ``|sum_n m_n u_n| / sqrt(sum_n m_n)``; 0 for a
                    COM-conserving mode, 1 for a rigid translation
                ``is_acoustic``: rigid translation of all atoms (Gamma only, else None)
                ``conserves_com``: COM residual below tolerance (Gamma only, else
                    None); acoustic modes never conserve the COM
        """
        u_cart = self.displacement_cartesian
        norm = float(np.sum(self.masses * np.sum(np.abs(u_cart) ** 2, axis=1)))
        result: dict[str, object] = {
            "is_gamma": self.is_gamma,
            "norm": norm,
            "com_residual": _com_residual(u_cart, self.masses),
            "is_acoustic": None,
            "conserves_com": None,
        }
        if self.is_gamma:
            mean = np.mean(u_cart, axis=0)
            spread = np.max(np.linalg.norm(u_cart - mean, axis=1))
            scale = np.max(np.linalg.norm(np.abs(u_cart), axis=1))
            result["is_acoustic"] = bool(scale > 0 and spread / scale < ACOUSTIC_TOL)
            result["conserves_com"] = bool(result["com_residual"] < COM_TOL)
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Convert the phonon mode to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "displacement": {
                "real": self.displacement.real.tolist(),
                "imag": self.displacement.imag.tolist(),
            },
            "masses": self.masses.tolist(),
            "lattice": self.lattice.tolist(),
            "atom_names": self.atom_names,
            "frequency": self.frequency,
            "q": self.q.tolist(),
            "source": self.source,
            "import_report": _to_jsonable(self.import_report),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PhononMode":
        """Create a PhononMode from a dictionary written by :meth:`to_dict`.

        Raises:
            ValueError: If the dictionary uses the legacy format (only ``vectors``),
                which needs crystal information to be converted
        """
        if "displacement" not in data:
            raise ValueError(
                "Legacy phonon format (only 'vectors'); convert with "
                "PhononsManager.add_mode(..., coords='reduced', kind='displacement')."
            )
        disp = data["displacement"]
        return cls(
            name=data["name"],
            displacement=np.array(disp["real"]) + 1j * np.array(disp["imag"]),
            masses=np.array(data["masses"]),
            lattice=np.array(data["lattice"]),
            atom_names=data.get("atom_names"),
            frequency=data.get("frequency"),
            q=np.array(data.get("q", [0.0, 0.0, 0.0])),
            source=data.get("source"),
            import_report=data.get("import_report"),
        )

    def __repr__(self) -> str:
        freq = f"{self.frequency:.3f} THz" if self.frequency is not None else None
        tags = []
        if self.consistency.get("is_acoustic"):
            tags.append("acoustic")
        if self.consistency.get("conserves_com") is False and not self.consistency.get("is_acoustic"):
            tags.append("COM violated")
        tag_str = f", {', '.join(tags)}" if tags else ""
        return (
            f"PhononMode(name='{self.name}', atoms={self.n_atoms}, freq={freq}, "
            f"q={self.q.tolist()}{tag_str})"
        )


def _com_residual(u_cart: np.ndarray, masses: np.ndarray) -> float:
    """Normalized COM residual ``|sum m u| / sqrt(sum m)`` for ``sum m |u|^2 = 1``."""
    return float(np.linalg.norm(np.sum(masses[:, np.newaxis] * u_cart, axis=0)) / np.sqrt(np.sum(masses)))


def _canonicalize(u_reduced: np.ndarray, masses: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Normalize to ``sum m |u_cart|^2 = 1`` and fix the global phase."""
    u_cart = u_reduced @ lattice
    norm = np.sqrt(np.sum(masses * np.sum(np.abs(u_cart) ** 2, axis=1)))
    if norm == 0:
        raise ValueError("Phonon vectors are all zero.")
    u_reduced = u_reduced / norm

    # Make the largest Cartesian component real and positive
    flat = (u_reduced @ lattice).ravel()
    largest = flat[np.argmax(np.abs(flat))]
    return u_reduced * (np.abs(largest) / largest)


def _to_jsonable(obj: object) -> object:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
