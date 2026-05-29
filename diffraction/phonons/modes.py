"""Phonon mode management."""

import numpy as np


class PhononMode:
    """Represents a phonon displacement mode.

    Attributes:
        name: Identifier for the phonon mode
        vectors: Displacement vectors (n_atoms x 3)
        atom_names: Optional list of atom names
        frequency: Mode frequency in THz or cm^-1 (optional)
    """

    def __init__(self, name: str, vectors: np.ndarray, atom_names: list[str] | None = None, frequency: float | None = None):
        """Initialize a phonon mode.

        Args:
            name: Identifier for the phonon mode
            vectors: Displacement vectors (n_atoms x 3)
            atom_names: Optional list of atom names for labeling
            frequency: Mode frequency in THz or cm^-1
        """
        self.name = name
        self.vectors = np.array(vectors)
        self.atom_names = atom_names
        self.frequency = frequency

    def __repr__(self) -> str:
        n_atoms = self.vectors.shape[0] if self.vectors.ndim > 1 else len(self.vectors) // 3
        return f"PhononMode(name='{self.name}', atoms={n_atoms}, freq={self.frequency})"


def load_phonon_from_dft(vectors: np.ndarray, atom_names: list[str] | None = None, mode_name: str = "dft_phonon") -> dict:
    """Load phonon displacement vectors from DFT calculation results.

    Args:
        vectors: Array of displacement vectors (n_atoms x 3 or n_modes x n_atoms x 3)
        atom_names: Optional list of atom names for labeling
        mode_name: Name identifier for the phonon mode

    Returns:
        dict: Phonon mode data ready to be added to a Crystal object

    Example:
        >>> from diffraction.phonons import load_phonon_from_dft
        >>> vectors = np.random.rand(4, 3) * 0.1
        >>> phonon_data = load_phonon_from_dft(vectors, atom_names=["Sr", "Ti", "O", "O"])
    """
    result = {"name": mode_name, "vectors": vectors}

    if atom_names is not None:
        if len(atom_names) != vectors.shape[0]:
            raise ValueError(
                f"atom_names length ({len(atom_names)}) must match number of atoms "
                f"({vectors.shape[0]})"
            )
        result["atom_names"] = atom_names

    return result


def create_phonon_mode(name: str, vectors: np.ndarray | list[list[float]], frequency: float | None = None) -> PhononMode:
    """Create a new phonon mode object.

    Args:
        name: Identifier for the phonon mode
        vectors: Displacement vectors (n_atoms x 3)
        frequency: Mode frequency in THz or cm^-1

    Returns:
        PhononMode: New phonon mode instance
    """
    return PhononMode(name=name, vectors=vectors, frequency=frequency)
