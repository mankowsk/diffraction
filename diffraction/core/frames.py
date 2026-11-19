"""Laboratory coordinate frames and conversions between them.

Three right-handed lab frames are used. They differ only by a cyclic permutation of
the axes, so every conversion is a proper rotation:

=========================  ==============  ==========  ===========
physical direction         diffcalc (You)  SwissFEL    SWIFT scene
=========================  ==============  ==========  ===========
up                         +x              +y          +z
along beam (downstream)    +y              +z          +x
left, looking downstream   +z              +x          +y
=========================  ==============  ==========  ===========

- ``"diffcalc"``: frame of diffcalc (H. You, J. Appl. Cryst. 32, 614 (1999)); the
  incident beam is +y, delta > 0 scatters up (+x), nu > 0 to the left (+z). All
  calculations are done in this frame.
- ``"swissfel"``: SwissFEL convention; default for user input and output.
- ``"swift"``: SWIFT visualization scene, which requires z to be vertical.

Convert only at the boundaries (user input/output, rendering) with :func:`convert`.
"""

import numpy as np

FRAMES = ("diffcalc", "swissfel", "swift")

# Rows: unit vectors (up, downstream, left) expressed in each frame
_PHYSICAL_AXES = {
    "diffcalc": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "swissfel": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "swift": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
}


def _check_frame(frame: str) -> None:
    if frame not in FRAMES:
        raise ValueError(f"Unknown frame '{frame}'; must be one of {FRAMES}")


def frame_matrix(from_frame: str, to_frame: str) -> np.ndarray:
    """Rotation matrix M with ``v_to = M @ v_from``.

    Args:
        from_frame: Frame the vector is given in
        to_frame: Frame to convert to

    Returns:
        ndarray: 3x3 orthogonal matrix with determinant +1
    """
    _check_frame(from_frame)
    _check_frame(to_frame)
    # v_from = P_from.T @ c and v_to = P_to.T @ c for physical components c
    return _PHYSICAL_AXES[to_frame].T @ _PHYSICAL_AXES[from_frame]


def convert(v: np.ndarray | list | tuple, from_frame: str, to_frame: str, kind: str = "vector") -> np.ndarray:
    """Convert vectors or rotation matrices between lab frames.

    Args:
        v: A 3-vector, an (N, 3) array of vectors, or a 3x3 rotation matrix
        from_frame: Frame ``v`` is given in (one of :data:`FRAMES`)
        to_frame: Frame to convert to
        kind: ``"vector"`` (default; also for (N, 3) arrays) or ``"rotation"`` for a 3x3
            operator R acting on vectors, converted as M R M^T

    Returns:
        ndarray: The converted vector(s) or matrix
    """
    M = frame_matrix(from_frame, to_frame)
    v = np.asarray(v, dtype=float)
    if kind == "rotation":
        if v.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {v.shape}")
        return M @ v @ M.T
    if kind != "vector":
        raise ValueError(f"kind must be 'vector' or 'rotation', got '{kind}'")
    if v.shape[-1] != 3:
        raise ValueError(f"Vectors must have 3 components, got shape {v.shape}")
    return v @ M.T
