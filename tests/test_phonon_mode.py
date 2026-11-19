"""Tests for PhononMode conventions, phonopy import and persistence."""

import warnings

import numpy as np
import pytest

from diffraction import Crystal
from diffraction.phonons import PhononConsistencyWarning, PhononMode, parse_phonopy_yaml

BAND_YAML = "tests/data/LiNbO3_band.yaml"
LINBO3_CIF = "diffraction/data/LiNbO3.cif"


@pytest.fixture
def linbo3():
    return Crystal.from_cif("LiNbO3", LINBO3_CIF)


@pytest.fixture
def imported(linbo3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhononConsistencyWarning)
        df = linbo3.phonons.add_from_dft(BAND_YAML)
    return linbo3, df


@pytest.fixture
def dft_data(imported):
    _, df = imported
    data = parse_phonopy_yaml(BAND_YAML)
    order = np.argsort(df.attrs["import_report"]["permutation"])
    return data, order


def test_import_gamma_modes(imported):
    crystal, df = imported
    assert len(df) == 30
    assert df.acoustic.tolist() == [True] * 3 + [False] * 27
    assert df.conserves_com[3:].all()
    assert not df.conserves_com[:3].any()


def test_import_atom_matching(imported):
    _, df = imported
    report = df.attrs["import_report"]
    assert report["species_match"]
    assert report["permutation"] == [0, 1, 2, 3, 4, 9, 8, 6, 5, 7]
    np.testing.assert_allclose(report["origin_shift"], [-0.0185] * 3, atol=1e-3)
    assert report["max_residual"] < 0.1


def test_import_warns_on_reordering(linbo3):
    with pytest.warns(PhononConsistencyWarning, match="reordered"):
        linbo3.phonons.add_from_dft(BAND_YAML, bands=[5])


def test_canonical_normalization(imported):
    crystal, df = imported
    for name in df.name:
        mode = crystal.phonons.get_mode(name)
        assert mode.consistency["norm"] == pytest.approx(1.0)
        np.testing.assert_allclose(np.sum(np.abs(mode.eigenvector) ** 2), 1.0)


def test_eigenvectors_orthonormal(imported):
    crystal, df = imported
    E = np.array([crystal.phonons.get_mode(n).eigenvector.ravel() for n in df.name])
    # Small deviations stem from the CIF lattice differing from the DFT lattice
    np.testing.assert_allclose(E.conj() @ E.T, np.eye(len(E)), atol=5e-3)


def test_user_conventions_agree(imported, dft_data):
    crystal, _ = imported
    data, order = dft_data
    e = data["qpoints"][0]["bands"][5]["eigenvector"][order].real
    masses = data["masses"][order]

    from_eigenvector = crystal.phonons.add_mode(
        "ev", e, coords="cartesian", kind="eigenvector", frame_lattice=data["lattice"], masses=masses
    )
    u_reduced = (e / np.sqrt(masses)[:, np.newaxis]) @ np.linalg.inv(data["lattice"])
    from_displacement = crystal.phonons.add_mode(
        "disp", 3.7 * u_reduced, coords="reduced", kind="displacement", masses=masses
    )

    np.testing.assert_allclose(from_eigenvector.displacement, from_displacement.displacement)
    np.testing.assert_allclose(
        from_eigenvector.displacement, crystal.phonons.get_mode("LiNbO3_band_q0_b5").displacement
    )


def test_eigenvector_as_displacement_warns(imported, dft_data):
    crystal, _ = imported
    data, order = dft_data
    e = data["qpoints"][0]["bands"][3]["eigenvector"][order].real
    e_reduced = e @ np.linalg.inv(data["lattice"])
    masses = data["masses"][order]

    with pytest.warns(PhononConsistencyWarning, match="kind='eigenvector'"):
        mode = crystal.phonons.add_mode("bad", e_reduced, masses=masses)
    assert mode.consistency["conserves_com"] is False

    with pytest.raises(ValueError, match="center of mass"):
        crystal.phonons.add_mode("bad", e_reduced, masses=masses, check="raise")


def test_finite_q_mode(imported):
    crystal, _ = imported
    df = crystal.phonons.add_from_dft(BAND_YAML, q_index=1, bands=[7])
    mode = crystal.phonons.get_mode(df.name[0])
    np.testing.assert_allclose(mode.q, [0.25, 0, 0])
    assert np.abs(mode.displacement.imag).max() > 0
    assert mode.consistency["conserves_com"] is None
    with pytest.raises(ValueError, match="Gamma"):
        crystal.calc_structure_factor_displaced((1, 0, 0), 9000, mode.name, 0.1)


def test_structure_factor_zero_amplitude(imported):
    crystal, _ = imported
    F0 = crystal.calc_structure_factor((1, 0, 0), 9000)
    F = crystal.calc_structure_factor_displaced((1, 0, 0), 9000, "LiNbO3_band_q0_b5", 0.0)
    assert F == pytest.approx(F0)
    F_array = crystal.calc_structure_factor_displaced((1, 0, 0), 9000, "LiNbO3_band_q0_b5", [0.0, 0.5])
    assert F_array.shape == (2,)


def test_amplitude_for_max_displacement(imported):
    crystal, _ = imported
    mode = crystal.phonons.get_mode("LiNbO3_band_q0_b5")
    Q = mode.amplitude_for_max_displacement(0.1)
    assert np.max(np.linalg.norm(Q * mode.real_displacement_cartesian(), axis=1)) == pytest.approx(0.1)


def test_dict_roundtrip(imported):
    crystal, _ = imported
    crystal.phonons.add_from_dft(BAND_YAML, q_index=1, bands=[7])
    for name in crystal.phonons.list_modes():
        mode = crystal.phonons.get_mode(name)
        restored = PhononMode.from_dict(mode.to_dict())
        np.testing.assert_allclose(restored.displacement, mode.displacement)
        np.testing.assert_allclose(restored.masses, mode.masses)
        np.testing.assert_allclose(restored.q, mode.q)
        assert restored.frequency == mode.frequency
        assert restored.import_report == mode.import_report


def test_save_load_roundtrip(imported, tmp_path):
    crystal, _ = imported
    crystal.save(str(tmp_path) + "/")
    loaded = Crystal.load("LiNbO3", str(tmp_path) + "/")
    # JSON is written with sorted keys, so the mode order is not preserved
    assert sorted(loaded.phonons.list_modes()) == sorted(crystal.phonons.list_modes())
    for name in crystal.phonons.list_modes():
        np.testing.assert_allclose(
            loaded.phonons.get_mode(name).displacement, crystal.phonons.get_mode(name).displacement
        )


def test_legacy_format_is_converted(linbo3, tmp_path):
    import json

    linbo3.save(str(tmp_path) + "/")
    json_path = tmp_path / "LiNbO3.json"
    data = json.loads(json_path.read_text())
    vectors = np.zeros((10, 3))
    vectors[2:4] = 0.01  # Nb along [111]
    data["phonons"] = {"legacy": {"name": "legacy", "vectors": vectors.tolist(), "atom_names": None, "frequency": 3.0}}
    json_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Legacy"):
        PhononMode.from_dict(data["phonons"]["legacy"])

    with pytest.warns(UserWarning, match="legacy format"):
        loaded = Crystal.load("LiNbO3", str(tmp_path) + "/")
    mode = loaded.phonons.get_mode("legacy")
    assert mode.frequency == 3.0
    assert mode.consistency["norm"] == pytest.approx(1.0)
    np.testing.assert_allclose(mode.displacement.real[2:4] / mode.displacement.real[2, 0], 1.0)


def test_animation_moves_all_periodic_images():
    pytest.importorskip("swift")
    pytest.importorskip("roboticstoolbox")
    from diffraction.visualization import CrystalVisualization

    crystal = Crystal.from_cif("SrTiO3", "diffraction/data/SrTiO3.cif")
    vis = CrystalVisualization(crystal, scale=0.1)
    vis.create_atoms()

    class StubEnv:
        def step(self, dt):
            pass

    vis.env = StubEnv()  # avoid launching a browser
    n_sites = len(set(vis.site_index.values()))
    assert len(vis.atoms) > n_sites  # boundary atoms have periodic images

    u = np.zeros((n_sites, 3))
    u[0] = [0.0, 0.0, 1.0]
    vis.animate(vectors=u, amplitudes=np.array([1.0]), dt=0)
    for atom_id, atom in vis.atoms.items():
        np.testing.assert_allclose(atom.q - atom.q0, 0.1 * u[vis.site_index[atom_id]], atol=1e-12)

    with pytest.raises(ValueError, match="lattice sites"):
        vis.animate(vectors=np.zeros((n_sites + 1, 3)), amplitudes=np.array([1.0]))
