"""DFT code importers for phonon modes."""

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
