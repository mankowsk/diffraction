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


def parse_yaml_phonons(yaml_path: str, mode_index: int = 0) -> dict:
    """Parse phonon data from a generic YAML file (e.g., PHONOPY disp.yaml).

    Args:
        yaml_path: Path to the YAML file containing phonon data
        mode_index: Index of the phonon mode to extract (default 0)

    Returns:
        dict: Phonon mode data with vectors and atom information
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for parsing YAML files. Install with: pip install pyyaml")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    atoms = data.get("atoms", []) or data.get("atom_names", [])
    vectors = data.get("force_displacement", []) or data.get("eigenvectors", [])

    if isinstance(vectors, list) and len(vectors) > 0:
        # Handle different YAML structures
        if isinstance(vectors[0], dict):
            # Nested structure - extract from first mode
            vectors = vectors[mode_index] if mode_index < len(vectors) else vectors[0]

    return {
        "name": f"mode_{mode_index}",
        "vectors": np.array(vectors),
        "atom_names": atoms if atoms else None,
    }
