"""Crystal.from_dft and phonon import into a different unit cell (rhombohedral DFT -> hexagonal)."""

import numpy as np
import pytest

from diffraction import Crystal
from diffraction.phonons.dft_importers import lattice_parameters, parse_phonopy_yaml

BAND = "tests/data/LiNbO3_band.yaml"
ENERGY = 9000
HKL_RH = [(1, 1, 1), (2, 1, 0), (1, 0, 0), (2, 2, 2), (3, 1, 1)]
OPTICAL = [3, 5, 10, 20, 29]  # a selection of Gamma-point optical bands


@pytest.fixture(scope="module")
def rhombohedral():
    return Crystal.from_dft("LNO_rh", BAND, cell="dft")


@pytest.fixture(scope="module")
def hexagonal():
    pytest.importorskip("spglib")
    return Crystal.from_dft("LNO_hex", BAND, cell="conventional")


def _mode(crystal, band):
    return crystal.phonons.get_mode(f"LiNbO3_band_q0_b{band}")


def _relative_sign(hexagonal, rhombohedral, band, M, mapping):
    """+1/-1: the overall sign of a mode is a convention and may differ between the cells."""
    u_hex = _mode(hexagonal, band).real_displacement_reduced()
    u_rh = (_mode(rhombohedral, band).real_displacement_reduced() @ np.linalg.inv(M))[mapping]
    sign = np.sign(np.sum(u_hex * u_rh))
    np.testing.assert_allclose(u_hex * np.sqrt(3), sign * u_rh, atol=1e-12)  # same pattern, 3x the atoms
    return sign


def _F_displaced(crystal, hkl, band, d=0.05):
    mode = _mode(crystal, band)
    return crystal.calc_structure_factor_displaced(hkl, ENERGY, mode.name, mode.amplitude_for_max_displacement(d))


def test_from_dft_cell(rhombohedral):
    data = parse_phonopy_yaml(BAND)
    assert rhombohedral.phonons.get_atom_names() == data["species"]
    params = lattice_parameters(data["lattice"])
    np.testing.assert_allclose([rhombohedral.unit_cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")], params)
    df = rhombohedral.phonons.last_import
    assert len(df) == 3 * len(data["species"])
    np.testing.assert_allclose(df["frequency_THz"], [b["frequency"] for b in data["qpoints"][0]["bands"]])
    assert df["acoustic"].sum() == 3 and df.loc[~df["acoustic"], "conserves_com"].all()


def test_conventional_cell(hexagonal, rhombohedral):
    assert hexagonal.space_group.startswith("R3c")
    assert len(hexagonal.phonons.get_atom_names()) == 30
    assert hexagonal.unit_cell["gamma"] == pytest.approx(120)
    report = hexagonal.phonons.last_import.attrs["import_report"]
    M = np.array(report["transformation"])
    assert round(np.linalg.det(M)) == 3 and report["max_residual"] < 1e-6
    assert hexagonal.phonons.last_import.loc[3:, "conserves_com"].all()

    signs = {band: _relative_sign(hexagonal, rhombohedral, band, M, np.array(report["mapping"])) for band in OPTICAL}
    for h_rh in HKL_RH:
        h_hex = tuple(int(v) for v in M @ h_rh)
        # Static: the hexagonal cell holds three rhombohedral cells (origin shift only changes the phase)
        assert abs(hexagonal.calc_structure_factor(h_hex, ENERGY)) == pytest.approx(
            3 * abs(rhombohedral.calc_structure_factor(h_rh, ENERGY)), rel=1e-9, abs=1e-8)
        # Same mode at the same maximum displacement
        for band in OPTICAL:
            assert abs(_F_displaced(hexagonal, h_hex, band)) == pytest.approx(
                3 * abs(_F_displaced(rhombohedral, h_rh, band, d=0.05 * signs[band])), rel=1e-8, abs=1e-8)

    # Reflections forbidden by the R centring (-h + k + l != 3n) stay absent when displaced
    for band in OPTICAL:
        assert abs(_F_displaced(hexagonal, (1, 0, 0), band)) < 1e-8
        assert abs(_F_displaced(hexagonal, (0, 0, 1), band)) < 1e-8


def test_import_into_existing_hexagonal_crystal(hexagonal, tmp_path):
    """Case: hexagonal crystal from a CIF, DFT phonons in the rhombohedral cell."""
    hexagonal.save(str(tmp_path) + "/")
    crystal = Crystal.from_cif("LNO_hex", str(tmp_path / "LNO_hex.cif"))
    df = crystal.phonons.add_from_dft(BAND)
    assert df.attrs["import_report"]["cell_ratio"] == 3
    # The CIF stores positions with 5 decimals, hence the tolerance
    for band in OPTICAL:
        np.testing.assert_allclose(_mode(crystal, band).displacement, _mode(hexagonal, band).displacement,
                                   rtol=1e-4, atol=1e-8)

    with pytest.raises(ValueError, match="Gamma"):
        crystal.phonons.add_from_dft(BAND, q_index=1)


def test_from_dft_save_load(rhombohedral, tmp_path):
    rhombohedral.save(str(tmp_path) + "/")
    loaded = Crystal.load("LNO_rh", str(tmp_path) + "/")
    assert sorted(loaded.phonons.list_modes()) == sorted(rhombohedral.phonons.list_modes())
    # Positions go through the CIF (5 decimals), hence the tolerance
    for band in OPTICAL:
        assert _F_displaced(loaded, (2, 1, 0), band) == pytest.approx(
            _F_displaced(rhombohedral, (2, 1, 0), band), rel=1e-4, abs=1e-6)


def test_structure_only():
    crystal = Crystal.from_dft("LNO", BAND, q_index=None)
    assert crystal.phonons.list_modes() == [] and crystal.phonons.last_import is None


IRREPS = "tests/data/irreps.yaml"  # synthetic fixture next to the band file, found by irreps="auto"


def test_irreps_labels(rhombohedral, hexagonal, tmp_path):
    from diffraction.phonons.dft_importers import parse_phonopy_irreps

    expected = {}
    for m in parse_phonopy_irreps(IRREPS)["modes"]:
        for b in m["bands"]:
            expected[b] = m["label"]
    for crystal in (rhombohedral, hexagonal):
        df = crystal.phonons.last_import
        assert df.attrs["import_report"]["point_group"] == "3m"
        labels = [None if isinstance(x, float) else x for x in df["irrep"]]  # pandas: None -> NaN
        assert labels == [expected[b] for b in df["band"]]
        assert _mode(crystal, 3).irrep == "E" and _mode(crystal, 4).irrep == "E"
    assert sum(label is None for label in expected.values()) == 2  # the set labelled 'None'

    rhombohedral.save(str(tmp_path) + "/")
    loaded = Crystal.load("LNO_rh", str(tmp_path) + "/")
    assert [loaded.phonons.get_mode(n).irrep for n in sorted(loaded.phonons.list_modes())] == \
           [rhombohedral.phonons.get_mode(n).irrep for n in sorted(rhombohedral.phonons.list_modes())]


def test_irreps_optional_and_checked(tmp_path):
    import shutil
    import warnings

    from diffraction.phonons.modes import PhononConsistencyWarning

    # No irreps.yaml next to the band file: no labels, no warning
    shutil.copy(BAND, tmp_path / "band.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhononConsistencyWarning)
        crystal = Crystal.from_dft("LNO", str(tmp_path / "band.yaml"))
    assert crystal.phonons.last_import["irrep"].isna().all()
    assert "point_group" not in crystal.phonons.last_import.attrs["import_report"]

    # Explicitly disabled
    crystal = Crystal.from_dft("LNO", BAND, irreps=None)
    assert crystal.phonons.last_import["irrep"].isna().all()

    # Wrong q-point: ignored with a warning
    text = open(IRREPS).read().replace("q-position: [    0.0000000,", "q-position: [    0.5000000,")
    (tmp_path / "irreps_q.yaml").write_text(text)
    with pytest.warns(PhononConsistencyWarning, match="q-point"):
        crystal = Crystal.from_dft("LNO", BAND, irreps=str(tmp_path / "irreps_q.yaml"))
    assert crystal.phonons.last_import["irrep"].isna().all()
