"""Tests for the lab-frame scattering geometry and orientation frames."""

import json

import numpy as np
import pytest

from diffraction import Crystal
from diffraction.core.frames import convert

BI2SE3_CIF = "tests/data/Bi2Se3.cif"


def _crystal(name, cif, frame="diffcalc", **constraints):
    crystal = Crystal.from_cif(name, cif)
    normal, along_beam = (0, 0, 1), (0, 1, 0)
    if frame != "diffcalc":
        normal, along_beam = (convert(v, "diffcalc", frame) for v in (normal, along_beam))
    crystal.add_orientation((0, 0, 1), tuple(normal), "normal", frame=frame)
    crystal.add_orientation((1, 0, 0), tuple(along_beam), "inplane along beam", frame=frame)
    crystal.calc_ub()
    crystal.set_constraints(**constraints)
    return crystal


CASES = [
    (("SrTiO3", "diffraction/data/SrTiO3.cif"), dict(eta=0, phi=0, delta=0), [(0, 0, 2), (1, 1, 1), (2, 1, 0)]),
    (("Bi2Se3", BI2SE3_CIF), dict(mu=0, chi=90, phi=60), [(0, 0, 15), (1, 0, 10), (0, 1, 5)]),
]


@pytest.mark.parametrize("cif,constraints,hkls", CASES)
def test_q_matches_ub(cif, constraints, hkls):
    crystal = _crystal(*cif, **constraints)
    energy = 7500
    UB = np.asarray(crystal.ubcalc.UB)
    for hkl in hkls:
        df = crystal.calc_angles(hkl, energy)
        for _, row in df.iterrows():
            geo = crystal.calc_lab_geometry(row, energy)
            np.testing.assert_allclose(geo.Q, geo.R_sample @ UB @ np.array(hkl), atol=1e-8)
            np.testing.assert_allclose(geo.hkl, hkl, atol=1e-8)
            np.testing.assert_allclose(np.linalg.norm(geo.k_f), np.linalg.norm(geo.k_i))


def test_lattice_and_frames():
    crystal = _crystal("Bi2Se3", BI2SE3_CIF, mu=0, chi=90, phi=60)
    geo = crystal.calc_lab_geometry({"delta": 20, "gamma": 5, "eta": 10, "chi": 80, "phi": 30}, 9000)
    UB = np.asarray(crystal.ubcalc.UB)
    # Real and reciprocal lattice: A^T (R UB) = 2 pi I
    np.testing.assert_allclose(geo.lattice.T @ geo.R_sample @ UB, 2 * np.pi * np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(geo.lattice, axis=0), [4.143, 4.143, 28.636], atol=1e-10)
    sf = geo.in_frame("swissfel")
    np.testing.assert_allclose(sf["k_i"] / np.linalg.norm(sf["k_i"]), [0, 0, 1])  # beam along SwissFEL z
    np.testing.assert_allclose(sf["Q"], sf["k_f"] - sf["k_i"], atol=1e-12)
    # Frame-independent scalars: Q . a = 2 pi h
    np.testing.assert_allclose(sf["lattice"].T @ sf["Q"], 2 * np.pi * geo.hkl, atol=1e-8)


def test_orientation_frames_equivalent():
    """The same physical orientation given in different frames yields the same UB."""
    ubs = [np.asarray(_crystal("SrTiO3", "diffraction/data/SrTiO3.cif", frame=f).ubcalc.UB)
           for f in ("diffcalc", "swissfel", "swift")]
    for ub in ubs[1:]:
        np.testing.assert_allclose(ub, ubs[0], atol=1e-12)


def test_orientation_json_round_trip_and_legacy(tmp_path):
    crystal = _crystal("SrTiO3", "diffraction/data/SrTiO3.cif", frame="swissfel")
    crystal.save(str(tmp_path) + "/")
    path = tmp_path / "SrTiO3.json"
    data = json.loads(path.read_text())
    assert all(o["frame"] == "swissfel" for o in data["orientations"])
    loaded = Crystal.load("SrTiO3", str(tmp_path) + "/")
    np.testing.assert_allclose(np.asarray(loaded.ubcalc.U), np.asarray(crystal.ubcalc.U))

    # Legacy file: no frame -> diffcalc, with a warning
    for o in data["orientations"]:
        o["xyz"] = list(convert(o["xyz"], "swissfel", "diffcalc"))
        del o["frame"]
    path.write_text(json.dumps(data))
    with pytest.warns(UserWarning, match="legacy"):
        legacy = Crystal.load("SrTiO3", str(tmp_path) + "/")
    assert all(o["frame"] == "diffcalc" for o in legacy.orientations)
    np.testing.assert_allclose(np.asarray(legacy.ubcalc.U), np.asarray(crystal.ubcalc.U), atol=1e-12)


def _sto_normal_up(kind):
    crystal = Crystal.from_cif("SrTiO3", "diffraction/data/SrTiO3.cif")
    crystal.add_orientation((0, 0, 1), (0, 1, 0), "normal up")  # SwissFEL frame (default)
    crystal.add_orientation((1, 0, 0), (0, 0, 1), "a along beam")
    crystal.calc_ub()
    if kind == "hkl":
        crystal.set_surface_normal((0, 0, 1))
    else:
        crystal.set_surface_normal((0, 1, 0), kind="lab")
    return crystal


@pytest.mark.parametrize("kind", ["hkl", "lab"])
def test_surface_normal(kind, tmp_path):
    crystal = _sto_normal_up(kind)
    geo = crystal.calc_lab_geometry({}, 9000)
    np.testing.assert_allclose(geo.in_frame("swissfel")["surface_normal"], [0, 1, 0], atol=1e-12)

    # diffcalc's incidence/exit angles agree with the drawn beams and surface
    crystal.set_constraints(mu=0, chi=0, phi=0)
    for _, row in crystal.calc_angles((1, 0, 3), 9000).iterrows():
        geo = crystal.calc_lab_geometry(row, 9000)
        k = np.linalg.norm(geo.k_i)
        n = geo.surface_normal
        np.testing.assert_allclose(np.degrees(np.arcsin(-geo.k_i @ n / k)), row["betain"], atol=1e-6)
        np.testing.assert_allclose(np.degrees(np.arcsin(geo.k_f @ n / k)), row["betaout"], atol=1e-6)

    crystal.save(str(tmp_path) + "/")
    loaded = Crystal.load("SrTiO3", str(tmp_path) + "/")
    assert loaded.surface_normal == crystal.surface_normal
    np.testing.assert_allclose(np.asarray(loaded.ubcalc.surf_nphi, dtype=float),
                               np.asarray(crystal.ubcalc.surf_nphi, dtype=float))


def test_reciprocal_axes_hexagonal():
    """Orientation hkl (1,0,0) along the beam puts a* (not a) along the beam; a is 30 deg off."""
    crystal = Crystal.from_cif("Bi2Se3", BI2SE3_CIF)
    crystal.add_orientation((0, 0, 1), (0, 0, 1), "normal", frame="diffcalc")
    crystal.add_orientation((1, 0, 0), (0, 1, 0), "inplane along beam", frame="diffcalc")
    crystal.calc_ub()
    geo = crystal.calc_lab_geometry({}, 7500)
    np.testing.assert_allclose(geo.reciprocal_lattice.T @ geo.lattice, 2 * np.pi * np.eye(3), atol=1e-10)

    def angle(u, v):
        return np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))

    assert angle(geo.reciprocal_lattice[:, 0], geo.k_i) == pytest.approx(0, abs=1e-6)
    assert angle(geo.lattice[:, 0], geo.k_i) == pytest.approx(30, abs=1e-6)
    sf = geo.in_frame("swissfel")
    np.testing.assert_allclose(sf["reciprocal_lattice"] @ geo.hkl, sf["Q"], atol=1e-10)


def test_calc_hkl_from_calc_angles_row():
    crystal = _crystal("SrTiO3", "diffraction/data/SrTiO3.cif", eta=0, phi=0, delta=0)
    df = crystal.calc_angles((1, 1, 1), 9000)
    for _, row in df.iterrows():
        np.testing.assert_allclose(crystal.calc_hkl(**row.to_dict(), energy=9000), (1, 1, 1), atol=1e-8)
