"""DFT code importers for phonon modes."""

import warnings

import numpy as np


def parse_vasp_phonons(vasp_output_path: str) -> dict:
    """Parse phonon displacement vectors from VASP PHONOPY output.

    Args:
        vasp_output_path: Path to VASP/PHONOPY output file (e.g., disp.yaml or LOCPOL)

    Returns:
        dict: Phonon mode data with vectors and atom information

    Note:
        This function is a placeholder for future implementation.
    """
    raise NotImplementedError(
        "VASP phonon parsing not yet implemented. Add your parsing logic here."
    )


def parse_quantumespresso_phonons(qe_output_path: str) -> dict:
    """Parse phonon displacement vectors from Quantum ESPRESSO output.

    Args:
        qe_output_path: Path to QE PHONON output file

    Returns:
        dict: Phonon mode data with vectors and atom information

    Note:
        This function is a placeholder for future implementation.
    """
    raise NotImplementedError(
        "Quantum ESPRESSO phonon parsing not yet implemented. Add your parsing logic here."
    )


def parse_abinit_phonons(abinit_output_path: str) -> dict:
    """Parse phonon displacement vectors from ABINIT output.

    Args:
        abinit_output_path: Path to ABINIT PHONON output file

    Returns:
        dict: Phonon mode data with vectors and atom information

    Note:
        This function is a placeholder for future implementation.
    """
    raise NotImplementedError(
        "ABINIT phonon parsing not yet implemented. Add your parsing logic here."
    )


def parse_phonopy_yaml(yaml_path: str) -> dict:
    """Parse a phonopy YAML output file containing eigenvectors.

    Supports ``band.yaml``, ``qpoints.yaml`` and ``mesh.yaml`` written with
    eigenvectors (phonopy option ``EIGENVECTORS = .TRUE.``). Phonopy eigenvectors
    are mass-weighted, normalized to 1 and given in Cartesian coordinates of the
    ``lattice`` stored in the file.

    Args:
        yaml_path: Path to the phonopy YAML file

    Returns:
        dict with keys:
            ``lattice``: real-space lattice (3 x 3, rows a1, a2, a3) in Angstrom
            ``species``: element symbols (n_atoms)
            ``positions``: reduced atomic positions (n_atoms x 3)
            ``masses``: atomic masses in amu (n_atoms)
            ``qpoints``: list of dicts with ``q`` (reduced) and ``bands``, a list of
                dicts with ``frequency`` (THz) and ``eigenvector`` (complex,
                n_atoms x 3, Cartesian)

    Raises:
        ImportError: If PyYAML is not installed
        ValueError: If the file contains no eigenvectors or lattice information
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for parsing YAML files. Install with: pip install pyyaml")

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=loader)

    if "lattice" not in data:
        raise ValueError(f"No 'lattice' found in {yaml_path}.")

    # Newer phonopy versions use 'points' with 'coordinates', older ones 'atoms' with 'position'
    if "points" in data:
        sites = data["points"]
        positions = [site["coordinates"] for site in sites]
    elif "atoms" in data:
        sites = data["atoms"]
        positions = [site["position"] for site in sites]
    else:
        raise ValueError(f"No atomic sites ('points' or 'atoms') found in {yaml_path}.")

    qpoints = []
    for qpoint in data.get("phonon", []):
        bands = []
        for band in qpoint["band"]:
            if "eigenvector" not in band:
                raise ValueError(
                    f"No eigenvectors in {yaml_path}. Rerun phonopy with EIGENVECTORS = .TRUE."
                )
            ev = np.array(band["eigenvector"], dtype=float)  # (n_atoms, 3, 2): real, imag
            bands.append(
                {
                    "frequency": float(band["frequency"]),
                    "eigenvector": ev[..., 0] + 1j * ev[..., 1],
                }
            )
        qpoints.append({"q": np.array(qpoint["q-position"], dtype=float), "bands": bands})

    return {
        "lattice": np.array(data["lattice"], dtype=float),
        "species": [site["symbol"] for site in sites],
        "positions": np.array(positions, dtype=float),
        "masses": np.array([site["mass"] for site in sites], dtype=float),
        "qpoints": qpoints,
    }


def parse_phonopy_irreps(yaml_path: str) -> dict:
    """Parse phonopy's ``irreps.yaml`` (``phonopy --irreps qx qy qz``).

    Args:
        yaml_path: Path to irreps.yaml

    Returns:
        dict with keys ``q`` (reduced), ``point_group`` (e.g. ``"-3m"``) and ``modes``: one
        dict per degenerate set with ``bands`` (0-based band indices), ``frequency`` (THz)
        and ``label`` (Mulliken symbol, None if phonopy could not assign one or away from
        Gamma, where phonopy writes no labels)
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for parsing YAML files. Install with: pip install pyyaml")

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=loader)
    if not isinstance(data, dict) or "normal_modes" not in data:
        raise ValueError(f"{yaml_path} is not a phonopy irreps.yaml (no 'normal_modes').")

    modes = []
    for entry in data["normal_modes"]:
        label = entry.get("ir_label")
        modes.append(
            {
                "bands": [int(i) - 1 for i in entry["band_indices"]],  # file is 1-based
                "frequency": float(entry["frequency"]),
                "label": None if label in (None, "None") else str(label),
            }
        )
    return {
        "q": np.array(data.get("q-position", [0.0, 0.0, 0.0]), dtype=float),
        "point_group": None if data.get("point_group") is None else str(data["point_group"]),
        "modes": modes,
    }


def lattice_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """(a, b, c, alpha, beta, gamma) in Angstrom and degrees from a lattice (rows a1, a2, a3)."""
    lattice = np.asarray(lattice, dtype=float)
    a, b, c = np.linalg.norm(lattice, axis=1)

    def angle(u, v):
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))))

    return float(a), float(b), float(c), angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]), angle(lattice[0], lattice[1])


def p1_lattice(lattice: np.ndarray, species: list[str], positions: np.ndarray):
    """xrayutilities ``SGLattice`` in space group P1 with all atoms listed explicitly.

    The atom order of ``base()`` equals the input order, so phonon vectors map 1:1.
    """
    import xrayutilities as xu

    atoms = []
    for symbol in species:
        element = getattr(xu.materials.elements, symbol.strip(), None)
        if element is None:
            raise ValueError(f"Unknown element '{symbol}' in the DFT structure.")
        atoms.append(element)
    pos = [("1a", tuple(float(x) for x in np.asarray(p) % 1.0)) for p in positions]
    return xu.materials.SGLattice(1, *lattice_parameters(lattice), atoms=atoms, pos=pos)


def conventional_cell(lattice: np.ndarray, species: list[str], positions: np.ndarray,
                      symprec: float = 1e-3) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    """Standardized conventional cell (spglib), e.g. hexagonal for a rhombohedral primitive cell.

    Returns:
        tuple: (lattice (rows), species, reduced positions, space group symbol with number)

    Raises:
        ImportError: If spglib is not installed
    """
    try:
        import spglib
    except ImportError as e:
        raise ImportError("cell='conventional' needs spglib: conda install -c conda-forge spglib") from e

    numbers = {s: i + 1 for i, s in enumerate(sorted(set(species)))}
    cell = (np.asarray(lattice, dtype=float), np.asarray(positions, dtype=float), [numbers[s] for s in species])
    with warnings.catch_warnings():  # spglib 2.x DeprecationWarning about its error handling
        warnings.simplefilter("ignore", DeprecationWarning)
        space_group = spglib.get_spacegroup(cell, symprec=symprec)
        standardized = spglib.standardize_cell(cell, to_primitive=False, symprec=symprec)
    if space_group is None or standardized is None:
        raise ValueError(f"spglib could not determine the symmetry (symprec={symprec}).")
    std_lattice, std_positions, std_numbers = standardized
    symbols = {n: s for s, n in numbers.items()}
    return np.asarray(std_lattice), [symbols[n] for n in std_numbers], np.asarray(std_positions), space_group


def parse_phonopy_born(path: str) -> dict:
    """Parse phonopy's ``BORN`` file.

    Format (phonopy docs): line 1 unit conversion factor or calculator name, line 2 the
    dielectric tensor epsilon_inf (xx xy xz yx yy yz zx zy zz), then one Born effective charge
    tensor (9 values, same order) per symmetry-independent atom. Z*_{gamma alpha}: gamma is the
    field/polarization direction, alpha the displacement direction (phonopy's NAC formula).
    Empty lines and lines starting with ``#`` are skipped.

    Returns:
        dict with ``factor`` (str), ``epsilon`` (3x3) and ``charges`` (n_independent x 3 x 3)
    """
    with open(path) as f:
        lines = [line.split("#")[0].strip() for line in f]
    lines = [line for line in lines if line]
    if len(lines) < 3:
        raise ValueError(f"{path}: a BORN file needs a factor line, epsilon and at least one Born charge.")

    def tensor(line: str, n: int) -> np.ndarray:
        values = [float(v) for v in line.split()]
        if len(values) != 9:
            raise ValueError(f"{path}, data line {n}: expected 9 values, got {len(values)}.")
        return np.array(values).reshape(3, 3)

    return {
        "factor": lines[0],
        "epsilon": tensor(lines[1], 2),
        "charges": np.array([tensor(line, i + 3) for i, line in enumerate(lines[2:])]),
    }


def expand_born_charges(charges: np.ndarray, lattice: np.ndarray, species: list[str], positions: np.ndarray,
                        symprec: float = 1e-3) -> np.ndarray:
    """Born charges for all atoms from those of the symmetry-independent atoms.

    The independent atoms are, as in phonopy, the sorted representatives of the spglib
    ``equivalent_atoms`` orbits. Atom j = (W, t) applied to independent atom i gets
    Z_j = R Z_i R^T with the Cartesian rotation R = A^T W A^-T (A: lattice rows).
    If ``charges`` already lists every atom it is returned unchanged.

    Raises:
        ValueError: If the number of tensors matches neither all nor the independent atoms
    """
    charges = np.asarray(charges, dtype=float)
    n_atoms = len(species)
    if len(charges) == n_atoms:
        return charges
    try:
        import spglib
    except ImportError as e:
        raise ImportError("Expanding Born charges of independent atoms needs spglib.") from e

    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    numbers = {s: i + 1 for i, s in enumerate(sorted(set(species)))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = spglib.get_symmetry_dataset((lattice, positions, [numbers[s] for s in species]), symprec=symprec)
    if dataset is None:
        raise ValueError(f"spglib could not determine the symmetry (symprec={symprec}).")
    independent = sorted(set(int(i) for i in dataset.equivalent_atoms))
    if len(charges) != len(independent):
        raise ValueError(
            f"BORN lists {len(charges)} Born charges; the structure has {n_atoms} atoms and "
            f"{len(independent)} symmetry-independent atoms {independent}."
        )

    A_T = lattice.T
    A_T_inv = np.linalg.inv(A_T)
    full = np.empty((n_atoms, 3, 3))
    done = np.zeros(n_atoms, dtype=bool)
    for Z, i in zip(charges, independent):
        for W, t in zip(dataset.rotations, dataset.translations):
            r = W @ positions[i] + t
            diff = positions - r
            diff -= np.round(diff)
            matches = np.where(np.linalg.norm(diff @ lattice, axis=1) < max(symprec, 1e-5) * 10)[0]
            for j in matches:
                if not done[j] and species[j] == species[i]:
                    R = A_T @ W @ A_T_inv
                    full[j] = R @ Z @ R.T
                    done[j] = True
    if not done.all():
        raise ValueError(f"Born charges could not be assigned to atoms {np.where(~done)[0].tolist()}.")
    return full
