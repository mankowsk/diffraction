"""Shared fixtures for tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_cif_path():
    """Path to a sample CIF file for testing."""
    return "diffraction/data/SrTiO3.cif"


@pytest.fixture
def sample_crystal(sample_cif_path):
    """Create a sample Crystal instance from CIF file."""
    from diffraction import Crystal
    return Crystal.from_cif("SrTiO3", sample_cif_path)


@pytest.fixture
def simple_lattice_params():
    """Simple cubic lattice parameters for testing."""
    return {
        "a": 4.0, "b": 4.0, "c": 4.0,
        "alpha": 90, "beta": 90, "gamma": 90
    }


@pytest.fixture
def orthorhombic_lattice():
    """Orthorhombic lattice parameters for testing."""
    return {
        "a": 5.2, "b": 7.4, "c": 9.1,
        "alpha": 90, "beta": 90, "gamma": 90
    }


@pytest.fixture
def sample_phonon_vectors():
    """Sample phonon displacement vectors for testing."""
    return np.array([
        [0.01, 0.02, -0.01],
        [-0.02, 0.01, 0.02],
        [0.015, -0.01, 0.01],
        [-0.01, 0.015, -0.015]
    ])


@pytest.fixture
def sample_ub_matrix():
    """Sample UB matrix for testing."""
    return np.array([
        [0.248, -0.003, 0.0],
        [0.0, 0.249, 0.0],
        [0.0, 0.0, 0.5]
    ])
