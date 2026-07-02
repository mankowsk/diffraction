"""Crystal class for X-ray diffraction calculations."""

import numpy as np
import pandas as pd
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub import calc as dccalc
import xrayutilities as xu


class Crystal:
    """Crystal class for X-ray diffraction calculations.

    This is the main API for working with crystal structures, UB matrix
    calculations, diffractometer angles, and phonon modes.

    Attributes:
        name: Name of the crystal material
        unit_cell: Dictionary containing lattice parameters (a, b, c, alpha, beta, gamma)
        ubcalc: diffcalc HklCalculation object for UB matrix computations
        orientations: List of added crystal orientations
        reflections: List of added reflection measurements
        u_matrix: Computed U matrix (3x3 numpy array)
        ub_matrix: Computed UB matrix (3x3 numpy array)
        xucalc: xrayutilities Crystal object
        xu_lattice: xrayutilities SGLattice object
        xu_cif_path: Path to the source CIF file
        phonons: PhononsManager instance for managing phonon modes
        constraints: Diffractometer angle constraints

    Example:
        >>> from diffraction import Crystal
        >>> crystal = Crystal.from_cif("SrTiO3", "data/SrTiO3.cif")
        >>> angles = crystal.calc_angles(0, 0, 1, 8048.0)
    """

    def __init__(self, name: str, a=1, b=1, c=1, alpha=90, beta=90, gamma=90):
        """Initialize a Crystal object with unit cell parameters.

        Args:
            name: Name of the crystal material
            a: Lattice parameter a (Angstroms)
            b: Lattice parameter b (Angstroms)
            c: Lattice parameter c (Angstroms)
            alpha: Lattice angle alpha (degrees)
            beta: Lattice angle beta (degrees)
            gamma: Lattice angle gamma (degrees)
        """
        self.name = name
        self.unit_cell = {
            "name": name, "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma
        }
        self.ubcalc = dccalc.UBCalculation("you")
        self.ubcalc.set_lattice(**self.unit_cell)
        self.orientations: list[dict] = []
        self.reflections: list[dict] = []
        self.u_matrix: np.ndarray | None = None
        self.ub_matrix: np.ndarray | None = None
        self.xucalc: xu.materials.Crystal | None = None
        self.xu_lattice: xu.materials.SGLattice | None = None
        self.xu_cif_path: str | None = None
        from diffraction.phonons import PhononsManager
        self.phonons = PhononsManager(self)
        self.constraints: dict[str, float | None] = {
            "mu": None, "eta": None, "chi": None, "phi": None,
            "delta": None, "gamma": None, "a_eq_b": None,
            "bin_eq_bout": None, "betain": None, "betaout": None,
            "qaz": None, "naz": None, "alpha": None, "beta": None,
            "bisect": None, "psi": None, "omega": None,
        }

    @classmethod
    def from_cif(cls, name: str, path_to_cif: str) -> "Crystal":
        """Create a Crystal from a CIF file.

        Args:
            name: Name identifier for the crystal
            path_to_cif: Path to the CIF file

        Returns:
            Crystal instance with unit cell parameters set from CIF data
        """
        xu_cif = xu.materials.cif.CIFFile(path_to_cif)
        xu_lattice = xu_cif.SGLattice()
        xucalc = xu.materials.Crystal(name, xu_lattice)
        unit_cell = {
            "name": name,
            "a": xucalc.a,
            "b": xucalc.b,
            "c": xucalc.c,
            "alpha": xucalc.alpha,
            "beta": xucalc.beta,
            "gamma": xucalc.gamma,
        }
        crystal = cls(**unit_cell)
        crystal.xucalc = xucalc
        crystal.xu_lattice = xu_lattice
        crystal.xu_cif_path = path_to_cif
        return crystal

    def add_orientation(self, hkl: tuple[int, int, int], xyz: tuple[float, float, float], tag: str | None = None) -> None:
        """Add a crystal orientation for UB matrix calculation.

        Args:
            hkl: Miller indices (h, k, l) of the reflection
            xyz: Normalized direction vector in crystal coordinates
            tag: Optional identifier for the orientation
        """
        self.orientations.append({"hkl": hkl, "xyz": xyz, "tag": tag})
        self.ubcalc.add_orientation(hkl, xyz, tag=tag)

    def add_reflection(self, hkl: tuple[int, int, int], position: Position, energy: float, tag: str | None = None) -> None:
        """Add a reflection measurement for UB matrix calculation.

        Args:
            hkl: Miller indices (h, k, l) of the reflection
            position: Position object containing diffractometer angles
            energy: X-ray energy in eV
            tag: Optional identifier for the reflection
        """
        self.reflections.append({"hkl": hkl, "position": position, "energy": energy, "tag": tag})
        self.ubcalc.add_reflection(hkl, position, energy, tag=tag)

    def set_constraints(self, **kwargs: float | None) -> None:
        """Set diffractometer angle constraints for calculations.

        Args:
            **kwargs: Constraint key-value pairs. Valid keys include:
                mu, eta, chi, phi, delta, gamma, a_eq_b, bin_eq_bout,
                betain, betaout, qaz, naz, alpha, beta, bisect, psi, omega
        """
        for k, v in kwargs.items():
            if k in self.constraints:
                self.constraints[k] = v

    def calc_ub(self, idx1: int = 0, idx2: int = 1) -> None:
        """Calculate UB matrix from two orientations or reflections.

        Args:
            idx1: Index of first orientation/reflection (default 0)
            idx2: Index of second orientation/reflection (default 1)
        """
        self.ubcalc.calc_ub(idx1 + 1, idx2 + 1)
        self.u_matrix = self.ubcalc.U
        self.ub_matrix = self.ubcalc.UB

    def fit_ub(self, indices: list[int] | None = None, refine_lattice: bool = False, refine_umatrix: bool = True) -> tuple[np.ndarray, dict]:
        """Refine UB matrix using reference reflections via least-squares fitting.

        Args:
            indices: List of reflection indices to use for refinement (default None)
            refine_lattice: Whether to refine lattice parameters (default False)
            refine_umatrix: Whether to refine U matrix (default True)

        Returns:
            tuple: (refined UB matrix, refined lattice parameters)
        """
        ub, lat = self.ubcalc.fit_ub(
            indices, refine_lattice=refine_lattice, refine_umatrix=refine_umatrix
        )
        self.u_matrix = self.ubcalc.U
        self.ub_matrix = self.ubcalc.UB
        return ub, lat

    def calc_angles(self, hkl: tuple[int, int, int], energy: float, pandas: bool = True) -> pd.DataFrame | list[dict]:
        """Calculate diffractometer angles for a given reflection.

        Args:
            hkl: Miller indices (h, k, l) of the reflection
            energy: X-ray energy in eV
            pandas: If True, return results as pandas DataFrame (default True)

        Returns:
            DataFrame or list of dicts: Diffractometer angle solutions for each configuration
        """
        constraints = Constraints({"nu" if k == "gamma" else k: v for k, v in self.constraints.items()})
        hklcalc = HklCalculation(self.ubcalc, constraints)
        lam = 12398.419843320025 / energy
        result = hklcalc.get_position(*hkl, lam)
        if pandas:
            result = pd.concat(
                [
                    pd.DataFrame.from_dict(
                        {
                            **{
                                "gamma" if k == "nu" else k: v
                                for k, v in tres[0].asdict.items()
                            },
                            **tres[1],
                        },
                        orient="index",
                        columns=[f"sol. {n}"],
                    )
                    for n, tres in enumerate(result)
                ],
                axis=1,
            ).T
        return result

    def calc_hkl(self, mu: float, delta: float, gamma: float, eta: float, chi: float, phi: float, energy: float) -> tuple[int, int, int] | None:
        """Calculate Miller indices from measured diffractometer angles.

        Args:
            mu: Mu angle in degrees
            delta: Delta angle in degrees
            gamma: Gamma (nu) angle in degrees
            eta: Eta angle in degrees
            chi: Chi angle in degrees
            phi: Phi angle in degrees
            energy: X-ray energy in eV

        Returns:
            tuple or None: Miller indices (h, k, l) or None if calculation fails
        """
        angs = [mu, delta, gamma, eta, chi, phi]
        pos = Position(*angs)
        lam = self.en2lam(energy)
        constraints = Constraints({"nu" if k == "gamma" else k: v for k, v in self.constraints.items()})
        hklcalc = HklCalculation(self.ubcalc, constraints)
        try:
            hkl = hklcalc.get_hkl(pos=pos, wavelength=lam)
        except Exception as e:
            print(str(e))
            return None
        return hkl

    def add_phonon(self, name: str, vectors: np.ndarray | list[list[float]], atom_names: list[str] | None = None, frequency: float | None = None) -> None:
        """Add a phonon displacement mode for structure factor calculations.

        Args:
            name: Identifier for the phonon mode
            vectors: Array of displacement vectors (shape: n_atoms x 3)
            atom_names: Optional list of atom names for labeling
            frequency: Mode frequency in THz or cm^-1

        Raises:
            ValueError: If vectors shape doesn't match number of atoms in crystal
        """
        self.phonons.add_mode(name=name, vectors=vectors, atom_names=atom_names, frequency=frequency)

    def delete_phonon(self, name: str) -> None:
        """Remove a phonon mode from the crystal.

        Args:
            name: Identifier of the phonon mode to remove
        """
        self.phonons.delete_mode(name)

    def calc_structure_factor(self, hkl: tuple[int, int, int], energy: float) -> complex:
        """Calculate the static structure factor F(h,k,l) at a given energy.

        Args:
            hkl: Miller indices (h, k, l)
            energy: X-ray energy in eV for form factor calculation

        Returns:
            complex: Structure factor amplitude

        Raises:
            ValueError: If xrayutilities Crystal or SGLattice is not initialized
        """
        if self.xucalc is None or self.xu_lattice is None:
            raise ValueError("xrayutilities Crystal and SGLattice must be set for structure factor calculation.")
        F = []
        for atompos in self.xucalc.lattice._wbase:
            u = atompos[1][1]
            q = self.xucalc.Q(hkl)
            f = atompos[0].f(np.linalg.norm(q), energy)
            F.append(f * np.exp(-2 * np.pi * 1j * np.dot(hkl, u)))
        return np.sum(np.array(F))

    def calc_structure_factor_displaced(self, hkl: tuple[int, int, int], energy: float, phonon_name: str, amplitude: np.ndarray | list[float] | float) -> complex | np.ndarray:
        """Calculate structure factor with atomic displacements from a phonon mode.

        Args:
            hkl: Miller indices (h, k, l)
            energy: X-ray energy in eV for form factor calculation
            phonon_name: Name of the phonon displacement mode
            amplitude: Displacement amplitude (scalar or array-like per atom)

        Returns:
            complex or ndarray: Structure factor amplitude(s) with displaced atoms

        Raises:
            ValueError: If phonon mode not found or xrayutilities objects not initialized
        """
        if phonon_name not in self.phonons._modes:
            raise ValueError(f"Phonon mode '{phonon_name}' not found.")
        if self.xucalc is None or self.xu_lattice is None:
            raise ValueError("xrayutilities Crystal and SGLattice must be set for structure factor calculation.")

        amplitude = np.asarray(amplitude)
        q = self.xucalc.Q(hkl)

        F = []
        for i, atompos in enumerate(self.xucalc.lattice._wbase):
            u = atompos[1][1]
            displaced_u = u + amplitude[:, np.newaxis] * self.phonons._modes[phonon_name].vectors[i]
            f = atompos[0].f(np.linalg.norm(q), energy)
            F.append(f * np.exp(-2 * np.pi * 1j * np.dot(hkl, displaced_u.T)).T)

        return np.sum(np.array(F), axis=0)

    def en2lam(self, en: float) -> float:
        """Convert X-ray energy to wavelength.

        Args:
            en: Energy in eV

        Returns:
            float: Wavelength in Angstroms
        """
        return 12398.419843320025 / en

    def lam2en(self, lam: float) -> float:
        """Convert wavelength to X-ray energy.

        Args:
            lam: Wavelength in Angstroms

        Returns:
            float: Energy in eV
        """
        return 12398.419843320025 / lam

    def show_unit_cell(self, distance: float = 0.0,  browser: str | None = None) -> object:
        """Display the crystal unit cell with reference planes.

        This method creates a visualization of the crystal structure without
        any phonon displacements, showing only the equilibrium atomic positions.

        Args:
            browser: which browser to open. For jupyter notebook or jupyterlab, use browser="notebook"
            distance: Distance for reference planes from origin (default 0.0)

        Raises:
            ValueError: If xucalc is not initialized (load from CIF first)
            ImportError: If visualization dependencies are not installed

        Returns:
            CrystalVisualization instance

        Example:
            >>> crystal = Crystal.from_cif("SrTiO3", "data/SrTiO3.cif")
            >>> vis = crystal.show_unit_cell()
        """
        if self.xucalc is None:
            raise ValueError(
                "Crystal must be loaded from CIF file to visualize. "
                "Use Crystal.from_cif('name', 'path/to/file.cif')."
            )

        from diffraction.visualization import CrystalVisualization

        vis = CrystalVisualization(self, scale=0.1)
        vis.create_atoms()
        vis.show(browser=browser)
        return vis

    def show_phonon(
        self,
        mode_name: str | None = None,
        scale: float = 1.0,
        dt: float = 0.05,
        amplitudes: np.ndarray | None = None,
    ) -> None:
        """Display and optionally animate a phonon mode.

        This method creates a visualization of the crystal with atomic
        displacements from the specified phonon mode. If amplitudes are
        provided, it will animate the mode over time.

        Args:
            mode_name: Name of the phonon mode to visualize (default None uses first mode)
            scale: Global scaling factor for visualization and displacements
            dt: Time step in seconds for animation (default 0.05)
            amplitudes: Array of amplitude values over time (n_modes x n_timepoints).
                       If None, only shows static view.

        Raises:
            ValueError: If xucalc is not initialized or no phonon modes registered
            KeyError: If the specified mode doesn't exist
            ImportError: If visualization dependencies are not installed

        Example:
            >>> # Show a single mode statically
            >>> crystal.show_phonon("soft_mode")
            >>>
            >>> # Animate with sinusoidal amplitudes
            >>> import numpy as np
            >>> n = np.linspace(0, 20*np.pi, 500)
            >>> amplitudes = 0.5 * np.array([np.sin(n), np.cos(n)])
            >>> crystal.show_phonon("soft_mode", amplitudes=amplitudes)
        """
        if self.xucalc is None:
            raise ValueError(
                "Crystal must be loaded from CIF file to visualize phonons. "
                "Use Crystal.from_cif('name', 'path/to/file.cif')."
            )

        self.phonons.show(mode_name=mode_name, scale=scale, dt=dt, amplitudes=amplitudes)

    def save(self, basepath: str = "diffraction/data/") -> None:
        """Save crystal configuration to the specified directory.

        Args:
            basepath: Base directory path for saving files (default "diffraction/data/")
        """
        from diffraction.io.json_io import save_crystal
        save_crystal(self, basepath)

    @classmethod
    def load(cls, name: str | None = None, basepath: str = "diffraction/data/") -> "Crystal":
        """Load a Crystal instance from saved JSON file.

        Args:
            name: Name of the crystal to load (default None lists available crystals)
            basepath: Base directory path for loading files (default "diffraction/data/")

        Returns:
            Crystal: Loaded crystal instance

        Raises:
            FileNotFoundError: If no name provided and no JSON files found in basepath
        """
        from diffraction.io.json_io import load_crystal
        return load_crystal(name, basepath)

    def __repr__(self) -> str:
        """Return a string representation of the crystal."""
        s = f"### LATTICE ###\n"
        for k, v in self.unit_cell.items():
            s += f"{k:6}: {v}\n"

        if self.xucalc is not None:
            s += f"\n### UNIT CELL ###\n"
            for i, atom in enumerate(self.xucalc.lattice._wbase):
                s += f"{i} {atom}\n"

        s += f"\n### ORIENTATIONS ###\n"
        s += f"{'hkl':15}{'xyz':15}{'tag'}\n"
        for orientation in self.orientations:
            s += f"{str(orientation['hkl']):15}{str(orientation['xyz']):15}{orientation['tag']}\n"

        s += f"\n### CONSTRAINTS ###\n"
        for k, v in self.constraints.items():
            if not v is None:
                s += f"{k:6}: {v}\n"

        # Add phonon modes section
        if self.phonons.list_modes():
            s += f"\n### PHONONS ###\n"
            for mode_name in self.phonons.list_modes():
                mode = self.phonons.get_mode(mode_name)
                freq_str = f", freq={mode.frequency}" if mode.frequency is not None else ""
                s += f"{mode_name}{freq_str}\n"

        return s
