"""Reciprocal space geometry utilities."""

import numpy as np


def reciprocal_lattice_vectors(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate reciprocal lattice vectors from direct lattice parameters.

    Args:
        a, b, c: Lattice constants in Angstroms
        alpha, beta, gamma: Lattice angles in degrees

    Returns:
        tuple: (a*, b*, c*) reciprocal lattice vectors as 3D arrays
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Volume of direct unit cell
    vol = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Reciprocal lattice constants
    a_star = (b * c * np.sin(alpha_rad)) / vol
    b_star = (a * c * np.sin(beta_rad)) / vol
    c_star = (a * b * np.sin(gamma_rad)) / vol

    # Reciprocal angles
    cos_alpha_star = (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad))
    cos_beta_star = (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad) * np.sin(gamma_rad))
    cos_gamma_star = (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad) * np.sin(beta_rad))

    alpha_star = np.degrees(np.arccos(cos_alpha_star))
    beta_star = np.degrees(np.arccos(cos_beta_star))
    gamma_star = np.degrees(np.arccos(cos_gamma_star))

    # Construct vectors in standard orientation (a along x, b in xy plane)
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    c_vec = np.array([
        c * np.cos(beta_rad),
        c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
        c * vol / (a * b * np.sin(gamma_rad))
    ])

    # Reciprocal vectors: a* = 2*pi * (b x c) / V, etc.
    a_star_vec = 2 * np.pi * np.cross(b_vec, c_vec) / vol
    b_star_vec = 2 * np.pi * np.cross(c_vec, a_vec) / vol
    c_star_vec = 2 * np.pi * np.cross(a_vec, b_vec) / vol

    return a_star_vec, b_star_vec, c_star_vec


def reciprocal_lattice_point(h: int, k: int, l: int, lattice_params: dict) -> np.ndarray:
    """Calculate the position of a reciprocal lattice point.

    Args:
        h, k, l: Miller indices
        lattice_params: Dictionary with 'a', 'b', 'c', 'alpha', 'beta', 'gamma'

    Returns:
        np.ndarray: Reciprocal lattice vector in Cartesian coordinates (Angstrom^-1)
    """
    a_star, b_star, c_star = reciprocal_lattice_vectors(
        lattice_params['a'], lattice_params['b'], lattice_params['c'],
        lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']
    )
    return h * a_star + k * b_star + l * c_star


def scattering_vector(q: np.ndarray) -> float:
    """Calculate the magnitude of a scattering vector.

    Args:
        q: Scattering vector in reciprocal space (Angstrom^-1)

    Returns:
        float: |q| in Angstrom^-1
    """
    return np.linalg.norm(q)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle between two vectors in degrees.

    Args:
        v1, v2: Input vectors

    Returns:
        float: Angle in degrees
    """
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
