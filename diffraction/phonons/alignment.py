"""Symmetry-adapted bases for degenerate phonon modes.

Eigenvectors of degenerate modes (e.g. the three T1u partners of a cubic perovskite) are
only defined up to a unitary rotation within their subspace; DFT codes return an arbitrary
basis. This module rotates such a set into partners polarized along orthonormal crystal
directions: x || a, z || a x b, y = z x x of the conventional cell (x, y, z for cubic cells;
a, in-plane normal to a and c for hexagonal/trigonal cells, also when the DFT used the
rhombohedral primitive cell). For every direction n the combination with the largest weight
of n-components (all atoms) is chosen; a set is aligned only if every partner is (almost)
purely polarized, i.e. the set transforms like a vector.
"""

import numpy as np

AXIS_NAMES = ("x", "y", "z")


def crystal_axes(lattice: np.ndarray) -> np.ndarray:
    """Orthonormal crystal frame (rows x, y, z) with x || a, z || a x b, in the frame of ``lattice``."""
    a, b = np.asarray(lattice, dtype=float)[:2]
    x = a / np.linalg.norm(a)
    z = np.cross(a, b)
    z /= np.linalg.norm(z)
    return np.array([x, np.cross(z, x), z])


def conventional_axes(lattice: np.ndarray, species: list[str], positions: np.ndarray,
                      symprec: float = 1e-3) -> np.ndarray:
    """:func:`crystal_axes` of the conventional cell (spglib), in the frame of ``lattice``.

    Falls back to the axes of ``lattice`` itself if spglib is not installed or fails.
    """
    import warnings

    try:
        import spglib
    except ImportError:
        return crystal_axes(lattice)
    numbers = {s: i + 1 for i, s in enumerate(sorted(set(species)))}
    cell = (np.asarray(lattice, dtype=float), np.asarray(positions, dtype=float), [numbers[s] for s in species])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return crystal_axes(lattice)
    # Conventional lattice (rows) in the input frame: std_lattice = input-frame lattice rotated by R
    conventional = np.asarray(dataset.std_lattice) @ np.asarray(dataset.std_rotation_matrix)
    return crystal_axes(conventional)


def degenerate_sets(frequencies: np.ndarray | list[float], tol: float = 1e-3) -> list[list[int]]:
    """Groups (>= 2 bands) of consecutive bands whose frequencies agree within ``tol`` (THz)."""
    freqs = np.asarray(frequencies, dtype=float)
    sets, i = [], 0
    while i < len(freqs):
        j = i + 1
        while j < len(freqs) and abs(freqs[j] - freqs[i]) < tol:
            j += 1
        if j - i > 1:
            sets.append(list(range(i, j)))
        i = j
    return sets


def _nearest_unitary(C: np.ndarray) -> np.ndarray | None:
    """Loewdin orthonormalization: the unitary matrix closest to C (None if C is singular)."""
    s, U = np.linalg.eigh(C.conj().T @ C)
    if s.min() < 1e-10:
        return None
    return C @ U @ np.diag(s ** -0.5) @ U.conj().T


def align_set(eigenvectors: np.ndarray, axes: np.ndarray, masses: np.ndarray | None = None,
              min_weight: float = 0.99):
    """Rotate one degenerate set into partners polarized along crystal axes.

    Two criteria are tried:

    1. ``"atoms"``: every atom moves along the partner's axis (exact for high site symmetry,
       e.g. all T1u/T2u sets of cubic perovskites, also non-polar ones);
    2. ``"dipole"``: the partner's dipole-like vector sum_n w_n e_n (w_n equal for equivalent
       atoms: 1 or 1/sqrt(m_n)) points along the axis. Such a sum transforms like a vector, so
       this picks the symmetry-adapted partners of polar sets (e.g. E modes of LiNbO3) even
       if individual atoms also move perpendicular to the axis.

    Args:
        eigenvectors: Mass-weighted eigenvectors of the set, shape (d, n_atoms, 3), Cartesian
        axes: Orthonormal directions (rows), e.g. from :func:`conventional_axes`
        masses: Atomic masses (amu), for the 1/sqrt(m) dipole weights
        min_weight: Minimum fraction of each partner (criterion 1: displacement, 2: dipole)
            along its axis; below, the next criterion is tried, finally the set is left unchanged

    Returns:
        tuple: (aligned eigenvectors (d, n_atoms, 3), axis index per partner, weight per
        partner, criterion ``"atoms"``/``"dipole"``), or None if the set cannot be aligned
    """
    E = np.asarray(eigenvectors, dtype=complex)
    d, n_atoms = E.shape[0], E.shape[1]
    axes = np.asarray(axes, dtype=float)
    V = E.reshape(d, -1).T  # (3 n_atoms, d)

    # 1. Every atom along the axis: top eigenvector of V^H P_n V per axis
    candidates = []
    for k, n in enumerate(axes):
        P = np.tile(n, n_atoms)
        w, c = np.linalg.eigh((V.conj().T * P) @ (V * P[:, None]))
        candidates.append((w[-1], k, c[:, -1]))
    chosen = sorted(sorted(candidates, key=lambda t: -t[0])[:d], key=lambda t: t[1])
    C = _nearest_unitary(np.column_stack([c for _, _, c in chosen]))
    if C is not None:
        partners, weights = [], []
        for j, (_, k, _) in enumerate(chosen):
            v = (V @ C[:, j]).reshape(-1, 3)
            along = v @ axes[k]
            partners.append(v * np.exp(-1j * np.angle(along[np.argmax(np.abs(along))])))
            weights.append(float(np.sum(np.abs(along) ** 2) / np.sum(np.abs(v) ** 2)))
        if min(weights) >= min_weight:
            return np.array(partners), [k for _, k, _ in chosen], weights, "atoms"

    # 2. Dipole-like vector along the axis
    weight_sets = [np.ones(n_atoms)]
    if masses is not None:
        weight_sets.append(1.0 / np.sqrt(np.asarray(masses, dtype=float)))
    best = None
    for w_atoms in weight_sets:
        def dipole(v: np.ndarray) -> np.ndarray:
            return w_atoms @ v  # (3,)

        D = np.array([axes @ dipole(V[:, j].reshape(-1, 3)) for j in range(d)]).T  # (3 axes, d)
        rows = sorted(np.argsort(-np.linalg.norm(D, axis=1))[:d])
        M = D[rows]
        if np.linalg.svd(M, compute_uv=False).min() < 1e-6:
            continue  # (nearly) non-polar for these weights
        C = _nearest_unitary(np.linalg.inv(M))
        if C is None:
            continue
        partners, weights = [], []
        for j, k in enumerate(rows):
            v = (V @ C[:, j]).reshape(-1, 3)
            p = dipole(v)
            v = v * np.exp(-1j * np.angle(p @ axes[k]))  # dipole along +axis
            partners.append(v)
            weights.append(float(np.abs(p @ axes[k]) ** 2 / np.sum(np.abs(p) ** 2)))
        if best is None or min(weights) > min(best[2]):
            best = (np.array(partners), list(rows), weights)
    if best is not None and min(best[2]) >= min_weight:
        return best[0], best[1], best[2], "dipole"
    return None


def align_degenerate_modes(frequencies, eigenvectors, axes: np.ndarray, masses: np.ndarray | None = None,
                           tol: float = 1e-3, min_weight: float = 0.99):
    """Align all degenerate sets of one q-point.

    Args:
        frequencies: Band frequencies (THz)
        eigenvectors: Mass-weighted eigenvectors per band, shape (n_bands, n_atoms, 3)
        axes: Orthonormal target directions (rows), in the frame of the eigenvectors, e.g.
            :func:`conventional_axes`
        masses: Atomic masses (amu), see :func:`align_set`
        tol: Frequency tolerance (THz) for degeneracy
        min_weight: See :func:`align_set`

    Returns:
        tuple: (eigenvectors with aligned sets, polarization label per band ("x"/"y"/"z" or
        None), weight per band (None if not aligned), list of aligned sets as dicts with
        ``bands`` and ``criterion``)
    """
    E = np.array(eigenvectors, dtype=complex)
    axes = np.asarray(axes, dtype=float)
    labels: list = [None] * len(E)
    weights: list = [None] * len(E)
    aligned_sets = []
    for bands in degenerate_sets(frequencies, tol):
        result = align_set(E[bands], axes, masses, min_weight)
        if result is None:
            continue
        vectors, axis_index, w, criterion = result
        for b, v, k, wk in zip(bands, vectors, axis_index, w):
            E[b] = v
            labels[b] = AXIS_NAMES[k]
            weights[b] = wk
        aligned_sets.append({"bands": bands, "criterion": criterion})
    return E, labels, weights, aligned_sets
