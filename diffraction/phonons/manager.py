"""Phonon mode management and visualization."""

import numpy as np
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal


@dataclass
class PhononMode:
    """Represents a phonon displacement mode.

    Attributes:
        name: Identifier for the phonon mode
        vectors: Displacement vectors (n_atoms x 3)
        atom_names: Optional list of atom names for labeling
        frequency: Mode frequency in THz or cm^-1 (optional)
    """
    name: str
    vectors: np.ndarray
    atom_names: list[str] | None = None
    frequency: float | None = None

    def __repr__(self) -> str:
        n_atoms = self.vectors.shape[0] if self.vectors.ndim > 1 else len(self.vectors) // 3
        return f"PhononMode(name='{self.name}', atoms={n_atoms}, freq={self.frequency})"


class PhononsManager:
    """Manages phonon modes for a Crystal instance.

    This class provides a centralized interface for storing, retrieving, and
    visualizing phonon displacement modes. It integrates with DFT importers
    and provides visualization capabilities.

    Attributes:
        crystal: Reference to the parent Crystal object
        _modes: Dictionary of PhononMode instances

    Example:
        >>> from diffraction import Crystal
        >>> crystal = Crystal.from_cif("SrTiO3", "data/SrTiO3.cif")
        >>> # Add a phonon mode
        >>> vectors = np.random.rand(5, 3) * 0.1
        >>> crystal.phonons.add_mode("test_mode", vectors)
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
        atom_names: list[str] | None = None,
        frequency: float | None = None,
    ) -> None:
        """Add a phonon displacement mode.

        Args:
            name: Identifier for the phonon mode
            vectors: Array of displacement vectors (shape: n_atoms x 3)
            atom_names: Optional list of atom names for labeling
            frequency: Mode frequency in THz or cm^-1

        Raises:
            ValueError: If vectors shape doesn't match number of atoms in crystal
        """
        vectors = np.array(vectors)
        expected_atoms = self._get_n_atoms()

        if vectors.shape[0] != expected_atoms:
            raise ValueError(
                f"Phonon mode '{name}' has {vectors.shape[0]} atoms, "
                f"but crystal has {expected_atoms} atoms."
            )

        self._modes[name] = PhononMode(
            name=name,
            vectors=vectors,
            atom_names=atom_names,
            frequency=frequency,
        )

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

    def get_atom_names(self) -> list[str] | None:
        """Get atom names from the crystal structure.

        Returns:
            list or None: List of element symbols for each atom, or None if unavailable
        """
        if self.crystal.xucalc is None:
            return None

        try:
            return [atom.name.symbol for atom in self.crystal.xucalc.lattice._wbase]
        except AttributeError:
            return None

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
            amplitudes: Either a callable that takes time array and returns amplitude values,
                       or an array of amplitude values over time. If None, uses a default sine modulation,
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
        else:
            mode = self.get_mode(mode_name)

        # Create visualization instance
        if vis is None:
            vis = CrystalVisualization(self.crystal, scale=scale)
            vis.create_atoms()
            vis.show(browser=browser)

        if amplitudes is None:
            n = np.linspace(0,20*np.pi,250)
            amplitudes = 0.5* np.array([np.sin(n), np.cos(n)])
            print(f"Showing default animation of phonon mode '{mode_name}'")

        # Prepare displacement vectors for animation
        n_atoms = self._get_n_atoms()
        mode_vectors = mode.vectors

        # Handle callable vs array amplitudes
        if callable(amplitudes):
            t = np.linspace(0, 1, 250)
            amplitude_values = amplitudes(t)
        else:
            amplitude_values = np.asarray(amplitudes)

        # Animate the mode
        vis.animate(vectors=mode_vectors, amplitudes=amplitude_values, dt=dt, scale=scale)
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

    def add_from_dft(
        self,
        vectors: np.ndarray,
        atom_names: list[str] | None = None,
        mode_name: str | None = None,
        frequency: float | None = None,
    ) -> PhononMode:
        """Add a phonon mode loaded from DFT calculation results.

        This is a convenience method that validates the vectors against the
        crystal structure and creates a new PhononMode instance.

        Args:
            vectors: Array of displacement vectors (n_atoms x 3 or n_modes x n_atoms x 3)
            atom_names: Optional list of atom names for labeling
            mode_name: Name identifier for the phonon mode (default None auto-generates)
            frequency: Mode frequency in THz or cm^-1

        Returns:
            PhononMode: The created phonon mode instance

        Raises:
            ValueError: If vectors shape doesn't match crystal atom count
        """
        if len(vectors.shape) == 3:
            raise ValueError(
                f"Expected 2D array (n_atoms x 3), got {vectors.shape}. "
                "For multiple modes, call add_from_dft separately for each mode."
            )

        expected_atoms = self._get_n_atoms()
        if vectors.shape[0] != expected_atoms:
            raise ValueError(
                f"Vectors have {vectors.shape[0]} atoms, but crystal has {expected_atoms} atoms."
            )

        # Auto-generate name if not provided
        if mode_name is None:
            import uuid
            mode_name = f"dft_phonon_{uuid.uuid4().hex[:8]}"

        self.add_mode(
            name=mode_name,
            vectors=vectors,
            atom_names=atom_names,
            frequency=frequency,
        )

        return self.get_mode(mode_name)


# Backward compatibility: expose PhononMode at package level
__all__ = ["PhononMode", "PhononsManager"]
