"""Schematic 4S+2D (You) diffractometer model for SWIFT.

The model is a URDF robot in the SWIFT frame (x along the beam, y left, z up; see
:mod:`diffraction.core.frames`), with the sample at the origin. It has two kinematic chains
from the fixed base:

- sample: mu -> eta -> chi -> phi
- detector: nu -> delta (arm and detector plate)

Joint axes and signs are taken from diffcalc's rotation matrices (converted to the SWIFT
frame), so that the joints can be set directly to the diffcalc angles. Dimensions are
schematic (scene units ~ metres) and can be changed via :func:`make_diffractometer_urdf`.
The default model is stored as ``models/diffractometer_4s2d.urdf``; regenerate it with
``python -m diffraction.visualization.diffractometer``.
"""

from pathlib import Path

import numpy as np

from diffraction.core.frames import convert
from diffraction.visualization.meshes import write_stl

MODELS_DIR = Path(__file__).resolve().parent / "models"
URDF_FILE = "diffractometer_4s2d.urdf"
CHI_ARC_FILE = "chi_arc.stl"

# Joint order of the robot's q vector
JOINTS = ("mu", "eta", "chi", "phi", "nu", "delta")

# Rotation axes in the diffcalc frame (H. You 1999, as implemented in diffcalc.hkl.geometry)
_AXES_DIFFCALC = {
    "mu": (1, 0, 0),
    "eta": (0, 0, -1),
    "chi": (0, 1, 0),
    "phi": (0, 0, -1),
    "nu": (1, 0, 0),
    "delta": (0, 0, -1),
}

DEFAULT_DIMENSIONS = {
    "base_height": 0.8,  # sample height above the floor plate
    "base_radius": 0.5,
    "nu_radius": 0.45,
    "mu_radius": 0.3,
    "eta_offset": 0.35,  # distance of the eta wheel from the sample (to the right, -y)
    "eta_radius": 0.22,
    "chi_radius": 0.2,
    "chi_width": 0.04,
    "phi_length": 0.18,  # holder surface ends 1 cm behind the sample surface
    "delta_offset": 0.5,  # distance of the delta wheel from the sample (to the right, -y)
    "delta_radius": 0.3,
    "detector_distance": 0.8,
    "detector_size": 0.2,
}

_COLORS = {
    "base": "0.35 0.35 0.38 1",
    "sample_stage": "0.75 0.75 0.78 1",
    "chi": "0.55 0.62 0.75 1",
    "detector_arm": "0.85 0.75 0.55 1",
    "detector": "0.2 0.2 0.2 1",
}


def joint_axis(name: str, frame: str = "swift") -> np.ndarray:
    """Rotation axis of a diffractometer joint (positive angle = right-handed about it)."""
    return convert(_AXES_DIFFCALC[name], "diffcalc", frame)


def _visual(xyz, rpy, geometry: str, color: str) -> str:
    return (
        f'    <visual>\n      <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>\n'
        f"      <geometry>{geometry}</geometry>\n"
        f'      <material name=""><color rgba="{color}"/></material>\n    </visual>\n'
    )


def _cyl(r: float, length: float) -> str:
    return f'<cylinder radius="{r:.4f}" length="{length:.4f}"/>'


def _box(x: float, y: float, z: float) -> str:
    return f'<box size="{x:.4f} {y:.4f} {z:.4f}"/>'


def _link(name: str, visuals: list[str]) -> str:
    return f'  <link name="{name}">\n' + "".join(visuals) + "  </link>\n"


def _joint(name: str, parent: str) -> str:
    ax = joint_axis(name)
    return (
        f'  <joint name="{name}" type="revolute">\n    <parent link="{parent}"/>\n    <child link="{name}"/>\n'
        f'    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <axis xyz="{ax[0]:g} {ax[1]:g} {ax[2]:g}"/>\n'
        f'    <limit lower="-6.2832" upper="6.2832" effort="1" velocity="1"/>\n  </joint>\n'
    )


def make_diffractometer_urdf(chi_arc_mesh: str | None = None, **dimensions: float) -> str:
    """URDF of the schematic 4S+2D diffractometer.

    Args:
        chi_arc_mesh: Path of the chi-arc STL (default: ``models/chi_arc.stl``)
        **dimensions: Overrides of :data:`DEFAULT_DIMENSIONS`

    Returns:
        str: URDF document
    """
    unknown = set(dimensions) - set(DEFAULT_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown dimensions {sorted(unknown)}; valid: {sorted(DEFAULT_DIMENSIONS)}")
    d = {**DEFAULT_DIMENSIONS, **dimensions}
    mesh = chi_arc_mesh or str(MODELS_DIR / CHI_ARC_FILE)
    h = d["base_height"]
    c = _COLORS
    half_pi = np.pi / 2

    links = [
        _link("base", [_visual((0, 0, -h + 0.03), (0, 0, 0), _cyl(d["base_radius"], 0.06), c["base"])]),
        # sample chain
        _link("mu", [
            _visual((0, 0, -h + 0.13), (0, 0, 0), _cyl(d["mu_radius"], 0.06), c["sample_stage"]),
            _visual((0, -d["eta_offset"] - 0.03, (-h + 0.16 - 0.04) / 2), (0, 0, 0),
                    _box(0.12, 0.06, h - 0.16 - 0.04), c["sample_stage"]),
        ]),
        _link("eta", [
            _visual((0, -d["eta_offset"], 0), (half_pi, 0, 0), _cyl(d["eta_radius"], 0.04), c["sample_stage"]),
            _visual((0, -(d["eta_offset"] + d["chi_radius"]) / 2, 0), (half_pi, 0, 0),
                    _cyl(0.04, d["eta_offset"] - d["chi_radius"]), c["sample_stage"]),
        ]),
        _link("chi", [
            _visual((0, 0, 0), (0, 0, 0), f'<mesh filename="{mesh}" scale="1 1 1"/>', c["chi"]),
        ]),
        _link("phi", [
            _visual((0, -d["chi_radius"] + d["phi_length"] / 2, 0), (half_pi, 0, 0),
                    _cyl(0.025, d["phi_length"]), c["sample_stage"]),
            _visual((0, -d["chi_radius"] + d["phi_length"] + 0.005, 0), (half_pi, 0, 0),
                    _cyl(0.045, 0.01), c["sample_stage"]),
        ]),
        # detector chain
        _link("nu", [
            _visual((0, 0, -h + 0.08), (0, 0, 0), _cyl(d["nu_radius"], 0.04), c["detector_arm"]),
            _visual((0, -d["delta_offset"] - 0.03, (-h + 0.1) / 2), (0, 0, 0),
                    _box(0.12, 0.06, h - 0.1), c["detector_arm"]),
        ]),
        _link("delta", [
            _visual((0, -d["delta_offset"], 0), (half_pi, 0, 0), _cyl(d["delta_radius"], 0.04), c["detector_arm"]),
            _visual((d["detector_distance"] / 2, -d["delta_offset"], 0), (0, half_pi, 0),
                    _cyl(0.025, d["detector_distance"]), c["detector_arm"]),
            _visual((d["detector_distance"], -d["delta_offset"] / 2, 0), (half_pi, 0, 0),
                    _cyl(0.02, d["delta_offset"]), c["detector_arm"]),
            _visual((d["detector_distance"] + 0.02, 0, 0), (0, 0, 0),
                    _box(0.03, d["detector_size"], d["detector_size"]), c["detector"]),
        ]),
    ]
    joints = [
        _joint("mu", "base"), _joint("eta", "mu"), _joint("chi", "eta"), _joint("phi", "chi"),
        _joint("nu", "base"), _joint("delta", "nu"),
    ]
    return '<?xml version="1.0"?>\n<robot name="diffractometer_4s2d">\n' + "".join(links) + "".join(joints) + "</robot>\n"


def chi_arc_triangles(radius: float, width: float, thickness: float = 0.02, n: int = 48) -> np.ndarray:
    """Triangles (N, 3, 3) of a half ring in the y-z plane around the x (chi) axis.

    The arc spans the right-hand side (-y), from the top through -y to the bottom, so the
    left side stays open; ``width`` is along x, ``thickness`` radial.
    """
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, n + 1)
    r_in, r_out = radius - thickness / 2, radius + thickness / 2
    tris = []

    def ring(r, x):
        return np.stack([np.full_like(t, x), r * np.cos(t), r * np.sin(t)], axis=1)

    corners = [ring(r_out, -width / 2), ring(r_out, width / 2), ring(r_in, width / 2), ring(r_in, -width / 2)]
    for a, b in zip(corners, corners[1:] + corners[:1]):  # four faces around the cross-section
        for i in range(n):
            tris += [(a[i], a[i + 1], b[i + 1]), (a[i], b[i + 1], b[i])]
    for idx in (0, n):  # end caps
        p = [ring_[idx] for ring_ in corners]
        tris += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    return np.array(tris)


def write_default_model() -> None:
    """Regenerate ``models/chi_arc.stl`` and ``models/diffractometer_4s2d.urdf``."""
    d = DEFAULT_DIMENSIONS
    write_stl(MODELS_DIR / CHI_ARC_FILE, chi_arc_triangles(d["chi_radius"], d["chi_width"]), b"diffraction chi arc")
    # Mesh referenced relative to the URDF so the file is portable
    (MODELS_DIR / URDF_FILE).write_text(make_diffractometer_urdf(chi_arc_mesh=CHI_ARC_FILE))


def sample_height(**dimensions: float) -> float:
    """Height of the sample (centre of rotation) above the SWIFT floor (z = 0)."""
    return {**DEFAULT_DIMENSIONS, **dimensions}["base_height"]


def load_diffractometer(**dimensions: float):
    """Diffractometer as ``rtb.Robot`` with q ordered as :data:`JOINTS` (radians).

    The robot base is raised by :func:`sample_height`, so the floor plate rests on the
    SWIFT floor and the sample sits at ``(0, 0, sample_height())``.

    Args:
        **dimensions: Overrides of :data:`DEFAULT_DIMENSIONS`; without overrides the
            committed ``models/diffractometer_4s2d.urdf`` is loaded
    """
    import tempfile

    import roboticstoolbox as rtb
    import spatialmath as sm

    if dimensions:
        tmp = Path(tempfile.mkdtemp()) / URDF_FILE
        tmp.write_text(make_diffractometer_urdf(**dimensions))
        tld, file = tmp.parent, tmp.name
    else:
        tld, file = MODELS_DIR, URDF_FILE
    links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(file, tld=str(tld))
    robot = rtb.Robot(links, name=name, urdf_string=urdf_string, urdf_filepath=urdf_filepath)
    robot.base = sm.SE3.Tz(sample_height(**dimensions))
    order = [robot.link_dict[j].jindex for j in JOINTS]
    if order != list(range(len(JOINTS))):
        raise RuntimeError(f"Unexpected joint order in {file}: {order}")
    return robot


if __name__ == "__main__":
    write_default_model()
    print(f"Wrote {MODELS_DIR / CHI_ARC_FILE} and {MODELS_DIR / URDF_FILE}")
