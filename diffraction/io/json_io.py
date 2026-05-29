"""JSON input/output operations for Crystal objects."""

import json
import os
import shutil
from typing import Any


def save_crystal(crystal: Any, basepath: str = "diffraction/data/") -> None:
    """Save a Crystal instance to disk as JSON with associated CIF file.

    Args:
        crystal: Crystal instance to save
        basepath: Base directory path for saving files (default "diffraction/data/")
    """
    path = basepath + crystal.name
    _save_to_disk(crystal, path)


def load_crystal(name: str | None = None, basepath: str = "diffraction/data/") -> Any:
    """Load a Crystal instance from saved JSON file.

    Args:
        name: Name of the crystal to load (default None lists available crystals)
        basepath: Base directory path for loading files (default "diffraction/data/")

    Returns:
        Crystal: Loaded crystal instance

    Raises:
        FileNotFoundError: If no name provided and no JSON files found in basepath
    """
    if name is not None:
        path = basepath + name + ".json"
    else:
        names = [n.split(".json")[0] for n in os.listdir(basepath) if ".json" in n and not n == ".json"]
        names.sort()
        raise FileNotFoundError(f"No crystal specified. Available crystals: {names}")

    return _load_from_disk(path)


def _save_to_disk(crystal: Any, path: str) -> None:
    """Persist crystal data to disk as JSON with associated CIF file.

    Args:
        crystal: Crystal instance to save
        path: Base path for saved files (without extension)
    """
    xu_cif_path = os.path.splitext(path)[0] + ".cif"
    json_path = os.path.splitext(path)[0] + ".json"

    data = _serialize_crystal(crystal)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    if crystal.xucalc is not None and crystal.xu_cif_path is not None:
        if not xu_cif_path == crystal.xu_cif_path:
            shutil.copy2(crystal.xu_cif_path, xu_cif_path)
            crystal.xu_cif_path = xu_cif_path
    elif crystal.xucalc is not None and crystal.xu_cif_path is None:
        crystal.xu_cif_path = xu_cif_path
        crystal.xucalc.toCIF(xu_cif_path)


def _serialize_crystal(crystal: Any) -> dict[str, Any]:
    """Serialize a Crystal instance to a dictionary.

    Args:
        crystal: Crystal instance to serialize

    Returns:
        dict: JSON-serializable representation of the crystal
    """
    return {
        "name": crystal.name,
        "unit_cell": crystal.unit_cell,
        "constraints": crystal.constraints,
        "orientations": crystal.orientations,
        "reflections": [
            {"hkl": r["hkl"], "position": list(r["position"].asdict.values()), "energy": r["energy"], "tag": r.get("tag")}
            for r in crystal.reflections
        ],
        "u_matrix": crystal.u_matrix.tolist() if crystal.u_matrix is not None else None,
        "ub_matrix": crystal.ub_matrix.tolist() if crystal.ub_matrix is not None else None,
        "xu_cif_path": crystal.xu_cif_path,
        "phonons": {k: v.tolist() for k, v in crystal.phonons.items()},
    }


def _load_from_disk(path: str) -> Any:
    """Deserialize crystal data from JSON file on disk.

    Args:
        path: Path to the JSON file containing crystal data

    Returns:
        Crystal: New Crystal instance with all properties restored
    """
    import xrayutilities as xu
    from diffcalc.hkl.geometry import Position

    with open(path, "r") as f:
        data = json.load(f)

    crystal = _instantiate_crystal(data)
    _restore_state(crystal, data)

    return crystal


def _instantiate_crystal(data: dict[str, Any]) -> Any:
    """Create a new Crystal instance from deserialized data.

    Args:
        data: Deserialized JSON data

    Returns:
        Crystal: New Crystal instance
    """
    from . import Crystal  # Avoid circular import

    crystal = Crystal(**data["unit_cell"])
    return crystal


def _restore_state(crystal: Any, data: dict[str, Any]) -> None:
    """Restore crystal state from deserialized data.

    Args:
        crystal: New Crystal instance
        data: Deserialized JSON data
    """
    import numpy as np
    from diffcalc.hkl.geometry import Position

    crystal.constraints = data["constraints"]
    for ori in data["orientations"]:
        crystal.add_orientation(ori["hkl"], ori["xyz"], ori.get("tag"))
    for refl in data["reflections"]:
        crystal.add_reflection(refl["hkl"], Position(*refl["position"]), refl["energy"], refl.get("tag"))

    if data["u_matrix"] is not None:
        crystal.u_matrix = np.array(data["u_matrix"])
        crystal.ubcalc.U = crystal.u_matrix
    if data["ub_matrix"] is not None:
        crystal.ub_matrix = np.array(data["ub_matrix"])
        crystal.ubcalc.UB = crystal.ub_matrix

    if data["xu_cif_path"] is not None and os.path.exists(data["xu_cif_path"]):
        xu_cif = xu.materials.cif.CIFFile(data["xu_cif_path"])
        xu_lattice = xu_cif.SGLattice()
        xucalc = xu.materials.Crystal(data["unit_cell"]["name"], xu_lattice)
        crystal.xucalc = xucalc
        crystal.xu_lattice = xu_lattice
        crystal.xu_cif_path = data["xu_cif_path"]

    if "phonons" in data:
        crystal.phonons = {k: np.array(v) for k, v in data["phonons"].items()}
