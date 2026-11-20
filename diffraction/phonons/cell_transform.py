"""Mapping between a DFT unit cell and a (super)cell of it, e.g. rhombohedral -> hexagonal.

Lattices are 3x3 arrays with rows a1, a2, a3. A crystal cell built from the DFT cell
satisfies, up to a rotation of the Cartesian frame,

    A_crystal = M @ A_dft        (M integer, det(M) = n_crystal / n_dft > 0)

so reduced coordinates transform as r_dft = r_crystal @ M and r_crystal = r_dft @ inv(M).
M is found from the metric tensors (A @ A.T), which do not depend on the Cartesian
convention (DFT codes and xrayutilities orient lattices differently).
"""

import itertools

import numpy as np


def find_cell_transformations(
    lattice_dft: np.ndarray, lattice_crystal: np.ndarray, max_index: int = 3, tol: float = 1e-3
) -> list[np.ndarray]:
    """All integer M (|entries| <= max_index, det > 0) with metric(M A_dft) = metric(A_crystal).

    Args:
        lattice_dft: DFT lattice (rows a1, a2, a3), Angstrom
        lattice_crystal: Crystal lattice (rows), Angstrom
        max_index: Largest absolute integer coefficient tried
        tol: Relative tolerance on the metric tensor entries

    Returns:
        list: Candidate matrices (3x3 int), smallest entries first; several candidates are
        related by lattice symmetry operations
    """
    A_dft = np.asarray(lattice_dft, dtype=float)
    G_dft = A_dft @ A_dft.T
    G_cr = np.asarray(lattice_crystal, dtype=float) @ np.asarray(lattice_crystal, dtype=float).T
    scale = np.max(np.abs(G_cr))

    vectors = np.array([n for n in itertools.product(range(-max_index, max_index + 1), repeat=3) if any(n)])
    lengths2 = np.einsum("ij,jk,ik->i", vectors, G_dft, vectors)
    rows = [vectors[np.abs(lengths2 - G_cr[i, i]) <= tol * scale] for i in range(3)]

    candidates = []
    for n0 in rows[0]:
        for n1 in rows[1]:
            if abs(n0 @ G_dft @ n1 - G_cr[0, 1]) > tol * scale:
                continue
            for n2 in rows[2]:
                if (abs(n0 @ G_dft @ n2 - G_cr[0, 2]) > tol * scale
                        or abs(n1 @ G_dft @ n2 - G_cr[1, 2]) > tol * scale):
                    continue
                M = np.array([n0, n1, n2])
                if round(np.linalg.det(M)) > 0:
                    candidates.append(M)
    candidates.sort(key=lambda M: (np.abs(M).sum(), -np.trace(M)))
    return candidates


def map_atoms(
    crystal_species: list[str],
    crystal_positions: np.ndarray,
    dft_species: list[str],
    dft_positions: np.ndarray,
    M: np.ndarray,
    lattice_dft: np.ndarray,
) -> dict[str, object]:
    """Assign every crystal atom to the DFT atom it is a lattice translate of.

    Crystal positions are expressed in DFT reduced coordinates (r @ M); a global origin
    shift is allowed (tried for every crystal atom of the rarest species and refined).

    Returns:
        dict with ``mapping`` (DFT index for each crystal atom), ``origin_shift`` (DFT
        reduced), ``residuals`` and ``max_residual`` (Angstrom), ``complete`` (every DFT
        atom has det(M) partners)
    """
    lattice_dft = np.asarray(lattice_dft, dtype=float)
    pos_cr = np.asarray(crystal_positions, dtype=float) @ np.asarray(M, dtype=float)
    pos_dft = np.asarray(dft_positions, dtype=float)
    ratio = round(abs(np.linalg.det(M)))

    def wrap(x: np.ndarray) -> np.ndarray:
        return x - np.round(x)

    def assign(shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mapping = np.empty(len(pos_cr), dtype=int)
        residuals = np.full(len(pos_cr), np.inf)
        for sp in set(crystal_species):
            i_cr = [i for i, s in enumerate(crystal_species) if s == sp]
            i_dft = [i for i, s in enumerate(dft_species) if s == sp]
            if not i_dft:
                continue
            diff = wrap(pos_cr[i_cr][:, None, :] - shift - pos_dft[i_dft][None, :, :])
            dist = np.linalg.norm(diff @ lattice_dft, axis=-1)
            best = dist.argmin(axis=1)
            for r, c in enumerate(best):
                mapping[i_cr[r]] = i_dft[c]
                residuals[i_cr[r]] = dist[r, c]
        return mapping, residuals

    rarest = min(set(dft_species), key=dft_species.count)
    i0 = dft_species.index(rarest)
    best = None
    for j, s in enumerate(crystal_species):
        if s != rarest:
            continue
        shift = wrap(pos_cr[j] - pos_dft[i0])
        mapping, residuals = assign(shift)
        if not np.all(np.isfinite(residuals)):
            continue
        shift = shift + np.mean(wrap(pos_cr - shift - pos_dft[mapping]), axis=0)
        mapping, residuals = assign(shift)
        if best is None or residuals.max() < best[2].max():
            best = (shift, mapping, residuals)

    if best is None:
        return {"mapping": None, "origin_shift": None, "residuals": None, "max_residual": np.inf, "complete": False}
    shift, mapping, residuals = best
    counts = np.bincount(mapping, minlength=len(dft_species))
    return {
        "mapping": mapping.tolist(),
        "origin_shift": shift.tolist(),
        "residuals": residuals.tolist(),
        "max_residual": float(residuals.max()),
        "complete": bool(np.all(counts == ratio)),
    }


def best_cell_mapping(
    crystal_species: list[str],
    crystal_positions: np.ndarray,
    lattice_crystal: np.ndarray,
    dft_species: list[str],
    dft_positions: np.ndarray,
    lattice_dft: np.ndarray,
    transformation: np.ndarray | None = None,
) -> dict[str, object]:
    """Find M and the atom mapping with the smallest residual.

    Args:
        transformation: Use this M instead of searching

    Returns:
        dict: :func:`map_atoms` result plus ``transformation`` (M as list) and
        ``n_candidates`` (number of lattice-equivalent M tried)

    Raises:
        ValueError: If no transformation maps the lattices and atoms onto each other
    """
    if transformation is not None:
        candidates = [np.asarray(transformation, dtype=int)]
    else:
        candidates = find_cell_transformations(lattice_dft, lattice_crystal)
    if not candidates:
        raise ValueError(
            "The crystal lattice is not an integer combination of the DFT lattice (|coefficients| <= 3); "
            "pass transformation=M explicitly."
        )
    best = None
    for M in candidates:
        result = map_atoms(crystal_species, crystal_positions, dft_species, dft_positions, M, lattice_dft)
        if result["complete"] and (best is None or result["max_residual"] < best["max_residual"] - 1e-9):
            best = {**result, "transformation": M.tolist()}
    if best is None:
        raise ValueError("No cell transformation maps every DFT atom onto the same number of crystal atoms.")
    best["n_candidates"] = len(candidates)
    return best
