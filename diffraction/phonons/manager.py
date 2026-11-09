"""Phonon mode management and visualization."""

import os
import warnings
import numpy as np
from typing import TYPE_CHECKING, Callable

from .modes import AMU_KG, PhononConsistencyWarning, PhononMode

if TYPE_CHECKING:
    import pandas as pd
    from diffraction.core.crystal import Crystal


class PhononsManager:
    """Manages phonon modes for a Crystal instance.

    This class provides a centralized interface for storing, retrieving, and
    visualizing phonon displacement modes. It integrates with DFT importers
    and provides visualization capabilities.

    All modes are stored as :class:`PhononMode` in canonical form
    (mass-normalized displacement in reduced coordinates of the crystal lattice).
    Phonon modes require a CIF with all atoms of the cell listed explicitly
    (space group P1), in the same order as the phonon vectors.

    Attributes:
        crystal: Reference to the parent Crystal object
        _modes: Dictionary of PhononMode instances

    Example:
        >>> from diffraction import Crystal
        >>> crystal = Crystal.from_cif("LiNbO3", "data/LiNbO3.cif")
        >>> # Import all Gamma-point modes from a phonopy calculation
        >>> summary = crystal.phonons.add_from_dft("band.yaml")
        >>> # Add a user-defined displacement pattern in reduced coordinates
        >>> crystal.phonons.add_mode("test_mode", vectors, coords="reduced", kind="displacement")
        >>> # Visualize the mode
        >>> crystal.show_phonon("test_mode")
    """

    def __init__(self, crystal: "Crystal"):
        """Initialize the PhononsManager with a reference to the Crystal.

        Args:
            crystal: The parent Crystal instance that owns these phonons
        """
        self.crystal = crystal
        self._modes: dict[str, PhononMode] = {}

    def add_mode(
        self,
        name: str,
        vectors: np.ndarray | list[list[float]],
        *,
        coords: str = "reduced",
        kind: str = "displacement",
        frame_lattice: np.ndarray | None = None,
        masses: np.ndarray | list[float] | None = None,
        atom_names: list[str] | None = None,
        frequency: float | None = None,
        q: np.ndarray | list[float] = (0.0, 0.0, 0.0),
        check: str = "warn",
    ) -> PhononMode:
        """Add a phonon mode given in any supported convention.

        The vectors are converted to the canonical form (mass-normalized
        displacement in reduced coordinates of the crystal lattice). Their
        overall scale is irrelevant.

        Args:
            name: Identifier for the phonon mode
            vectors: Array of vectors (shape: n_atoms x 3), real or complex
            coords: ``"reduced"`` (units of the lattice vectors) or ``"cartesian"`` (Angstrom)
            kind: ``"displacement"`` (physical displacement pattern) or
                ``"eigenvector"`` (mass-weighted eigenvector, ``e = sqrt(m) u``)
            frame_lattice: Lattice (rows a1, a2, a3) defining the Cartesian frame
                of the input; defaults to the crystal lattice
            masses: Atomic masses in amu; defaults to the masses of the crystal atoms
            atom_names: Element symbols; defaults to those of the crystal atoms
            frequency: Mode frequency in THz
            q: Wave vector in reduced reciprocal coordinates
            check: ``"warn"``, ``"raise"`` or ``"off"`` if a Gamma-point optical
                mode does not conserve the center of mass

        Returns:
            PhononMode: The added mode

        Raises:
            ValueError: If vectors shape doesn't match number of atoms in crystal
        """
        vectors = np.asarray(vectors)
        expected_atoms = self._get_n_atoms()

        if vectors.shape[0] != expected_atoms:
            raise ValueError(
                f"Phonon mode '{name}' has {vectors.shape[0]} atoms, "
                f"but crystal has {expected_atoms} atoms."
            )

        mode = PhononMode.from_vectors(
            name,
            vectors,
            self.get_masses() if masses is None else masses,
            self.get_lattice(),
            coords=coords,
            kind=kind,
            frame_lattice=frame_lattice,
            q=q,
            frequency=frequency,
            atom_names=self.get_atom_names() if atom_names is None else atom_names,
            check=check,
        )
        self.add(mode)
        return mode

    def add(self, mode: PhononMode) -> None:
        """Register an existing PhononMode under its name.

        Args:
            mode: Phonon mode in canonical form

        Raises:
            ValueError: If the mode's atom count doesn't match the crystal
        """
        expected_atoms = self._get_n_atoms()
        if mode.n_atoms != expected_atoms:
            raise ValueError(
                f"Phonon mode '{mode.name}' has {mode.n_atoms} atoms, "
                f"but crystal has {expected_atoms} atoms."
            )
        self._modes[mode.name] = mode

    def get_mode(self, name: str) -> PhononMode:
        """Retrieve a phonon mode by name.

        Args:
            name: Name of the phonon mode

        Returns:
            PhononMode: The requested phonon mode

        Raises:
            KeyError: If the phonon mode doesn't exist
        """
        if name not in self._modes:
            raise KeyError(f"Phonon mode '{name}' not found. Available modes: {list(self._modes.keys())}")
        return self._modes[name]

    def delete_mode(self, name: str) -> None:
        """Remove a phonon mode from the manager.

        Args:
            name: Name of the phonon mode to remove
        """
        if name in self._modes:
            del self._modes[name]

    def list_modes(self) -> list[str]:
        """List all available phonon mode names.

        Returns:
            list: Names of all registered phonon modes
        """
        return list(self._modes.keys())

    def get_modes_dict(self) -> dict[str, dict]:
        """Get all phonon modes as JSON-serializable dictionaries.

        Returns:
            dict: Mapping of mode names to their dictionary representation
        """
        return {name: mode.to_dict() for name, mode in self._modes.items()}

    def get_atom_names(self) -> list[str] | None:
        """Get atom names from the crystal structure.

        Returns:
            list or None: List of element symbols for each atom, or None if unavailable
        """
        if self.crystal.xucalc is None:
            return None

        return [site[0].basename for site in self.crystal.xucalc.lattice._wbase]

    def _get_n_atoms(self) -> int:
        """Get the number of atoms in the crystal unit cell.

        Returns:
            int: Number of unique atomic sites

        Raises:
            ValueError: If xucalc is not initialized
        """
        if self.crystal.xucalc is None:
            raise ValueError("Crystal must be loaded from CIF file to determine atom count.")
        return len(self.crystal.xucalc.lattice._wbase)

    def get_lattice(self) -> np.ndarray:
        """Real-space lattice of the crystal (rows a1, a2, a3) in Angstrom.

        Raises:
            ValueError: If xucalc is not initialized
        """
        if self.crystal.xucalc is None:
            raise ValueError("Crystal must be loaded from CIF file to determine the lattice.")
        xucalc = self.crystal.xucalc
        return np.array([xucalc.a1, xucalc.a2, xucalc.a3], dtype=float)

    def get_masses(self) -> np.ndarray:
        """Atomic masses of the crystal atoms in amu.

        Raises:
            ValueError: If xucalc is not initialized
        """
        if self.crystal.xucalc is None:
            raise ValueError("Crystal must be loaded from CIF file to determine atomic masses.")
        return np.array([atom[0].weight / AMU_KG for atom in self.crystal.xucalc.lattice._wbase])

    def _get_sites(self) -> tuple[list[str], np.ndarray]:
        """Element symbols and reduced positions of all atoms in the cell.

        Raises:
            ValueError: If the CIF does not list all atoms explicitly (non-P1)
        """
        sites = list(self.crystal.xucalc.lattice.base())
        if len(sites) != self._get_n_atoms():
            raise ValueError(
                "Phonon import requires a CIF listing all atoms of the cell explicitly (P1); "
                f"found {self._get_n_atoms()} unique sites but {len(sites)} atoms."
            )
        species = [site[0].name for site in sites]
        positions = np.array([site[1] for site in sites], dtype=float)
        return species, positions

    def show(
        self,
        mode_name: str | None = None,
        scale: float = 1.0,
        dt: float = 0.05,
        amplitudes: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        browser: str | None = None,
        vis: str | None = None,
    ) -> object:
        """Display and optionally animate a phonon mode.

        This method creates a visualization of the crystal with atomic
        displacements from the specified phonon mode. If amplitudes are
        provided, it will animate the mode over time.

        Args:
            mode_name: Name of the phonon mode to visualize (default None uses first mode)
            scale: Global scaling factor for visualization and displacements
            dt: Time step in seconds for animation (default 0.05)
            amplitudes: Normal coordinate Q (Angstrom * sqrt(amu)) over time, either as a callable
                       that takes a time array or as an array. Use
                       ``mode.amplitude_for_max_displacement(d)`` to convert from a maximum atomic
                       displacement d in Angstrom. If None, uses a sine modulation with 0.2 Angstrom
                       maximum displacement,
                       If 0, only shows static view.
            browser: which browser to open. For jupyter notebook or jupyterlab, use browser="notebook" else None
            vis: a visualisation instance of the crystal unit cell. If provided, the phonon will be shown in 
                       the given existing visualisation.


        Raises:
            KeyError: If the specified mode doesn't exist
            ImportError: If visualization dependencies (SWIFT, Robotics Toolbox) are not installed
            ValueError: If no phonon modes are registered

        Returns:
            CrystalVisualization instance

        Example:
            >>> # Show a single mode statically
            >>> vcrystal.phononsis = .show("soft_mode")
            >>>
            >>> # Animate with sinusoidal amplitudes
            >>> import numpy as np
            >>> n = np.linspace(0, 20*np.pi, 500)
            >>> amplitudes = 0.5 * np.array([np.sin(n), np.cos(n)])
            >>> vis = crystal.phonons.show("soft_mode", amplitudes=amplitudes)
        """
        from diffraction.visualization import CrystalVisualization

        # Select mode to visualize
        if mode_name is None:
            if not self._modes:
                raise ValueError("No phonon modes registered. Add a mode first.")
            mode_name = next(iter(self._modes))
            print(f"Showing default mode: {mode_name}")
        mode = self.get_mode(mode_name)

        # Create visualization instance
        if vis is None:
            vis = CrystalVisualization(self.crystal, scale=scale)
            vis.create_atoms()
            vis.show(browser=browser)

        if amplitudes is None:
            n = np.linspace(0,20*np.pi,250)
            amplitudes = mode.amplitude_for_max_displacement(0.2) * np.sin(n)
            print(f"Showing default animation of phonon mode '{mode_name}'")

        # Cartesian displacement per unit normal coordinate (Angstrom / (Angstrom sqrt(amu)))
        mode_vectors = mode.real_displacement_cartesian()

        # Handle callable vs array amplitudes
        if callable(amplitudes):
            t = np.linspace(0, 1, 250)
            amplitude_values = amplitudes(t)
        else:
            amplitude_values = np.asarray(amplitudes)

        # Animate the mode
        vis.animate(vectors=mode_vectors, amplitudes=amplitude_values, dt=dt)
        return vis

    def show_unit_cell(self, distance: float = 0.0,  browser: str | None = None) -> object:
        """Display the crystal unit cell with reference planes.

        This method creates a visualization of the crystal structure without
        any phonon displacements, showing only the equilibrium atomic positions.

        Args:
            distance: Distance for reference planes from origin (default 0.0)
            browser: which browser to open. For jupyter notebook or jupyterlab, use browser="notebook" else None

        Raises:
            ValueError: If xucalc is not initialized (load from CIF first)
            ImportError: If visualization dependencies are not installed

        Returns:
            CrystalVisualization instance

        Example:
            >>> crystal = Crystal.from_cif("SrTiO3", "data/SrTiO3.cif")
            >>> vis = crystal.show_unit_cell()
        """
        if self.crystal.xucalc is None:
            raise ValueError(
                "Crystal must be loaded from CIF file to visualize. "
                "Use Crystal.from_cif('name', 'path/to/file.cif')."
            )

        from diffraction.visualization import CrystalVisualization

        vis = CrystalVisualization(self.crystal, scale=0.1)
        vis.create_atoms()
        vis.show(browser=browser)
        return vis

    def add_from_dft(
        self,
        path: str,
        *,
        q_index: int = 0,
        bands: list[int] | None = None,
        name_prefix: str | None = None,
        position_tol: float = 0.1,
    ) -> "pd.DataFrame":
        """Import phonon modes from a phonopy YAML file (band.yaml, qpoints.yaml, mesh.yaml).

        The import never fails on physics checks. Instead, atoms are matched to
        the CIF by species and position (allowing a global origin shift) and
        reordered if necessary, and every mode carries a consistency report.
        Acoustic modes are imported as well; they are flagged and naturally do
        not conserve the center of mass.

        Args:
            path: Path to the phonopy YAML file (written with eigenvectors)
            q_index: Index of the q-point in the file
            bands: Band indices to import (default: all)
            name_prefix: Prefix for mode names (default ``"<file stem>_q<q_index>_"``);
                modes are named ``<prefix>b<band>``
            position_tol: Atom matching residual (Angstrom) above which a warning is issued

        Returns:
            pandas.DataFrame: One row per imported mode with frequency and
            consistency checks. The atom matching report is in ``df.attrs["import_report"]``.

        Raises:
            ValueError: If the number of atoms differs from the crystal
        """
        import pandas as pd
        from .dft_importers import parse_phonopy_yaml

        data = parse_phonopy_yaml(path)
        qpoint = data["qpoints"][q_index]
        cif_species, cif_positions = self._get_sites()
        lattice = self.get_lattice()

        if len(data["species"]) != len(cif_species):
            raise ValueError(
                f"{path} has {len(data['species'])} atoms, but crystal has {len(cif_species)} atoms."
            )

        report = _match_atoms(data["species"], data["positions"], cif_species, cif_positions, lattice)
        report["lattice_mismatch"] = _lattice_mismatch(data["lattice"], lattice)
        report["masses_dft"] = data["masses"].tolist()
        report["masses_cif"] = self.get_masses().tolist()
        report["file"] = os.path.abspath(path)

        if not report["species_match"]:
            warnings.warn(
                f"Atomic species in {path} do not match the crystal; modes imported in DFT atom order.",
                PhononConsistencyWarning,
                stacklevel=2,
            )
        elif report["max_residual"] > position_tol:
            warnings.warn(
                f"Atom positions in {path} deviate from the crystal by up to "
                f"{report['max_residual']:.3f} Angstrom after matching.",
                PhononConsistencyWarning,
                stacklevel=2,
            )
        if not report["identity"]:
            warnings.warn(
                f"Atoms in {path} were reordered to match the crystal: {report['permutation']}",
                PhononConsistencyWarning,
                stacklevel=2,
            )

        # perm[i] is the crystal index of DFT atom i; invert to reorder DFT data
        order = np.argsort(report["permutation"])
        masses = data["masses"][order]
        atom_names = [data["species"][i] for i in order]

        if name_prefix is None:
            name_prefix = f"{os.path.splitext(os.path.basename(path))[0]}_q{q_index}_"
        band_indices = range(len(qpoint["bands"])) if bands is None else bands

        rows = []
        for b in band_indices:
            band = qpoint["bands"][b]
            mode = PhononMode.from_vectors(
                f"{name_prefix}b{b}",
                band["eigenvector"][order],
                masses,
                lattice,
                coords="cartesian",
                kind="eigenvector",
                frame_lattice=data["lattice"],
                q=qpoint["q"],
                frequency=band["frequency"],
                atom_names=atom_names,
                source=f"phonopy:{os.path.basename(path)} q{q_index} b{b}",
                import_report=report,
                check="off",
            )
            self.add(mode)
            rows.append(
                {
                    "name": mode.name,
                    "band": b,
                    "frequency_THz": mode.frequency,
                    "acoustic": mode.consistency["is_acoustic"],
                    "com_residual": mode.consistency["com_residual"],
                    "conserves_com": mode.consistency["conserves_com"],
                }
            )

        df = pd.DataFrame(rows)
        df.attrs["import_report"] = report
        return df


def _match_atoms(
    dft_species: list[str],
    dft_positions: np.ndarray,
    cif_species: list[str],
    cif_positions: np.ndarray,
    lattice: np.ndarray,
) -> dict[str, object]:
    """Match DFT atoms to crystal atoms by species and position.

    Tries every global origin shift that maps an atom of the rarest species onto
    an atom of the same species, assigns atoms per species by minimum distance
    (periodic, measured in Angstrom with ``lattice``) and keeps the best shift.

    Returns:
        dict with ``species_match``, ``permutation`` (crystal index of each DFT
        atom), ``identity``, ``origin_shift`` (reduced), ``residuals`` and
        ``max_residual`` (Angstrom)
    """
    from scipy.optimize import linear_sum_assignment

    n = len(dft_species)
    if sorted(dft_species) != sorted(cif_species):
        return {
            "species_match": False,
            "permutation": list(range(n)),
            "identity": True,
            "origin_shift": [0.0, 0.0, 0.0],
            "residuals": None,
            "max_residual": None,
        }

    def wrap(x: np.ndarray) -> np.ndarray:
        return x - np.round(x)

    def assign(shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        perm = np.empty(n, dtype=int)
        residuals = np.empty(n)
        for sp in set(dft_species):
            i_dft = [i for i, s in enumerate(dft_species) if s == sp]
            i_cif = [i for i, s in enumerate(cif_species) if s == sp]
            diff = wrap(cif_positions[i_cif][np.newaxis, :, :] - dft_positions[i_dft][:, np.newaxis, :] - shift)
            dist = np.linalg.norm(diff @ lattice, axis=-1)
            rows, cols = linear_sum_assignment(dist)
            for r, c in zip(rows, cols):
                perm[i_dft[r]] = i_cif[c]
                residuals[i_dft[r]] = dist[r, c]
        return perm, residuals

    rarest = min(set(dft_species), key=dft_species.count)
    i0 = dft_species.index(rarest)
    candidates = [wrap(cif_positions[j] - dft_positions[i0]) for j, s in enumerate(cif_species) if s == rarest]
    best = None
    for shift in candidates:
        perm, residuals = assign(shift)
        # Refine the shift as the mean offset of all matched atoms
        shift = shift + np.mean(wrap(cif_positions[perm] - dft_positions - shift), axis=0)
        perm, residuals = assign(shift)
        if best is None or residuals.max() < best[2].max():
            best = (shift, perm, residuals)

    shift, perm, residuals = best
    return {
        "species_match": True,
        "permutation": perm.tolist(),
        "identity": bool(np.all(perm == np.arange(n))),
        "origin_shift": shift.tolist(),
        "residuals": residuals.tolist(),
        "max_residual": float(residuals.max()),
    }


def _lattice_mismatch(lattice_a: np.ndarray, lattice_b: np.ndarray) -> dict[str, list[float]]:
    """Relative length and angle (degree) differences between two lattices."""

    def params(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.linalg.norm(lat, axis=1)
        angles = np.array(
            [
                np.degrees(np.arccos(np.dot(lat[j], lat[k]) / (lengths[j] * lengths[k])))
                for j, k in ((1, 2), (0, 2), (0, 1))
            ]
        )
        return lengths, angles

    len_a, ang_a = params(np.asarray(lattice_a))
    len_b, ang_b = params(np.asarray(lattice_b))
    return {
        "relative_length_difference": ((len_a - len_b) / len_b).tolist(),
        "angle_difference_deg": (ang_a - ang_b).tolist(),
    }
