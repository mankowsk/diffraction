"""The schematic diffractometer URDF reproduces diffcalc's rotations."""

import numpy as np
import pytest

pytest.importorskip("roboticstoolbox")

from diffcalc.hkl.geometry import Position, get_rotation_matrices  # noqa: E402

from diffraction.core.frames import convert  # noqa: E402
from diffraction.visualization.diffractometer import JOINTS, load_diffractometer  # noqa: E402


def test_urdf_kinematics_match_diffcalc():
    robot = load_diffractometer()
    rng = np.random.default_rng(3)
    for _ in range(20):
        angles = dict(zip(JOINTS, rng.uniform(-180, 180, size=len(JOINTS))))
        q = np.radians([angles[j] for j in JOINTS])
        pos = Position(mu=angles["mu"], delta=angles["delta"], nu=angles["nu"],
                       eta=angles["eta"], chi=angles["chi"], phi=angles["phi"])
        MU, DELTA, NU, ETA, CHI, PHI = get_rotation_matrices(Position.asradians(pos))

        R_phi = robot.fkine(q, end="phi").R
        R_delta = robot.fkine(q, end="delta").R
        np.testing.assert_allclose(R_phi, convert(MU @ ETA @ CHI @ PHI, "diffcalc", "swift", kind="rotation"), atol=1e-10)
        np.testing.assert_allclose(R_delta, convert(NU @ DELTA, "diffcalc", "swift", kind="rotation"), atol=1e-10)


def test_custom_dimensions():
    robot = load_diffractometer(detector_distance=1.2)
    assert robot.n == len(JOINTS)
    with pytest.raises(ValueError):
        load_diffractometer(arm_length=1.0)
