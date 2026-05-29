"""UB matrix calculation utilities."""

import numpy as np
from diffcalc.ub import calc as dccalc


class UBMatrixCalculator:
    """Utility class for UB matrix operations.

    This module provides helper functions for UB matrix manipulation,
    validation, and analysis separate from the main Crystal class.
    """

    @staticmethod
    def validate_ub_matrix(ub: np.ndarray) -> dict[str, float]:
        """Validate a UB matrix and compute diagnostic metrics.

        Args:
            ub: 3x3 UB matrix to validate

        Returns:
            dict: Diagnostic metrics including determinant, condition number, etc.
        """
        det = np.linalg.det(ub)
        cond = np.linalg.cond(ub)
        u_norm = np.linalg.norm(ub[:3, :3] if ub.shape[0] == 4 else ub)

        return {
            "determinant": float(det),
            "condition_number": float(cond),
            "norm": float(u_norm),
            "is_valid": bool(np.isfinite(det) and cond < 1e10),
        }

    @staticmethod
    def extract_u_matrix(ub: np.ndarray) -> np.ndarray:
        """Extract the U matrix from a UB matrix.

        Args:
            ub: 3x3 or 4x4 UB matrix

        Returns:
            np.ndarray: 3x3 U matrix (rotation component)
        """
        if ub.shape[0] == 4:
            return ub[:3, :3]
        return ub.copy()

    @staticmethod
    def extract_b_matrix(ub: np.ndarray, lattice_params: dict) -> np.ndarray:
        """Extract the B matrix from a UB matrix given lattice parameters.

        Args:
            ub: 3x3 or 4x4 UB matrix
            lattice_params: Dictionary with 'a', 'b', 'c', 'alpha', 'beta', 'gamma'

        Returns:
            np.ndarray: 3x3 B matrix (reciprocal lattice component)
        """
        u = UBMatrixCalculator.extract_u_matrix(ub)
        b_matrix = np.linalg.inv(u) @ ub if ub.shape[0] == 4 else np.eye(3)
        return b_matrix

    @staticmethod
    def rotate_ub_matrix(ub: np.ndarray, rotation_axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
        """Apply a rotation to the UB matrix.

        Args:
            ub: Current UB matrix
            rotation_axis: Axis of rotation (x, y, z) as unit vector
            angle_deg: Rotation angle in degrees

        Returns:
            np.ndarray: Rotated UB matrix
        """
        from scipy.spatial.transform import Rotation

        axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
        rot = Rotation.from_rotvec(angle_deg * np.pi / 180 * axis)
        R = rot.as_matrix()

        if ub.shape[0] == 4:
            ub_new = ub.copy()
            ub_new[:3, :3] = R @ ub[:3, :3]
            return ub_new
        return R @ ub
