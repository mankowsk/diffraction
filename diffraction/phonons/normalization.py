"""Phonon mode normalization and filtering utilities."""

import numpy as np


def normalize_phonon_mode(vectors: np.ndarray, target_amplitude: float = 1.0) -> np.ndarray:
    """Normalize phonon displacement vectors to a target amplitude.

    Args:
        vectors: Displacement vectors (n_atoms x 3)
        target_amplitude: Target RMS amplitude for normalization

    Returns:
        np.ndarray: Normalized displacement vectors

    Raises:
        ValueError: If the input vectors have zero amplitude
    """
    rms = np.sqrt(np.mean(vectors**2))
    if rms == 0:
        raise ValueError("Cannot normalize zero-amplitude phonon mode")
    return vectors * (target_amplitude / rms)


def filter_phonon_by_frequency(
    vectors: np.ndarray, frequencies: np.ndarray, freq_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Filter phonon modes by frequency threshold.

    Args:
        vectors: Array of displacement vectors (n_modes x n_atoms x 3)
        frequencies: Array of mode frequencies (n_modes,)
        freq_threshold: Frequency cutoff in THz or cm^-1

    Returns:
        tuple: (filtered_vectors, filtered_frequencies)
    """
    mask = np.abs(frequencies) >= freq_threshold
    return vectors[mask], frequencies[mask]


def remove_acoustic_modes(vectors: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Remove acoustic modes from a set of phonon modes.

    Acoustic modes have near-zero frequency and correspond to rigid translations
    or rotations of the crystal.

    Args:
        vectors: Array of displacement vectors (n_modes x n_atoms x 3)
        tol: Tolerance for identifying zero-amplitude modes

    Returns:
        np.ndarray: Filtered array with acoustic modes removed
    """
    # Calculate mode amplitudes
    amplitudes = np.sqrt(np.mean(vectors**2, axis=(1, 2)))
    mask = amplitudes > tol
    return vectors[mask]


def project_out_acoustic_modes(vectors: np.ndarray) -> np.ndarray:
    """Project out acoustic (translational/rotational) components from phonon modes.

    Args:
        vectors: Array of displacement vectors (n_modes x n_atoms x 3)

    Returns:
        np.ndarray: Projected displacement vectors with acoustic components removed
    """
    n_modes, n_atoms = vectors.shape[:2]
    projected = np.zeros_like(vectors)

    for i in range(n_modes):
        mode = vectors[i].flatten()
        # Simple projection: remove mean displacement (translational component)
        mean_disp = np.mean(mode.reshape(n_atoms, 3), axis=0)
        projected[i] = vectors[i] - mean_disp

    return projected
