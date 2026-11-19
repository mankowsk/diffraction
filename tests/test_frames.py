"""Tests for the lab coordinate frames (diffraction/core/frames.py)."""

import itertools

import numpy as np
import pytest

from diffraction.core.frames import FRAMES, convert, frame_matrix

# (up, downstream, left) in each frame, as in the table of the module docstring
ANCHORS = {
    "diffcalc": {"up": [1, 0, 0], "downstream": [0, 1, 0], "left": [0, 0, 1]},
    "swissfel": {"up": [0, 1, 0], "downstream": [0, 0, 1], "left": [1, 0, 0]},
    "swift": {"up": [0, 0, 1], "downstream": [1, 0, 0], "left": [0, 1, 0]},
}


@pytest.mark.parametrize("src,dst", list(itertools.product(FRAMES, FRAMES)))
def test_anchors(src, dst):
    for direction in ("up", "downstream", "left"):
        np.testing.assert_allclose(convert(ANCHORS[src][direction], src, dst), ANCHORS[dst][direction])


@pytest.mark.parametrize("src,dst", list(itertools.product(FRAMES, FRAMES)))
def test_proper_rotation_and_round_trip(src, dst):
    M = frame_matrix(src, dst)
    np.testing.assert_allclose(M @ M.T, np.eye(3))
    assert np.isclose(np.linalg.det(M), 1.0)
    v = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(convert(convert(v, src, dst), dst, src), v)


@pytest.mark.parametrize("frame", FRAMES)
def test_frames_right_handed(frame):
    a = ANCHORS[frame]
    # left = up x downstream in every right-handed frame
    np.testing.assert_allclose(np.cross(a["up"], a["downstream"]), a["left"])


def test_rotation_conversion_consistent_with_vectors():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q * np.sign(np.linalg.det(q))
    v = rng.normal(size=3)
    R_sf = convert(R, "diffcalc", "swissfel", kind="rotation")
    np.testing.assert_allclose(R_sf @ convert(v, "diffcalc", "swissfel"), convert(R @ v, "diffcalc", "swissfel"))


def test_diffcalc_conventions():
    """The table matches diffcalc: beam along +y, delta up, nu left."""
    from diffcalc.hkl.geometry import Position, get_rotation_matrices

    def kf(**angles):
        pos = {**dict(mu=0, delta=0, nu=0, eta=0, chi=0, phi=0), **angles}
        _, DELTA, NU, *_ = get_rotation_matrices(Position.asradians(Position(**pos)))
        return NU @ DELTA @ np.array(ANCHORS["diffcalc"]["downstream"], dtype=float)

    up, left = (np.array(ANCHORS["diffcalc"][d]) for d in ("up", "left"))
    assert kf(delta=30) @ up > 0.49
    assert kf(nu=30) @ left > 0.49


def test_unknown_frame():
    with pytest.raises(ValueError):
        convert([1, 0, 0], "diffcalc", "lab")
