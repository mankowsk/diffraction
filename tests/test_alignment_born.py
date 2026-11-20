"""Alignment of degenerate modes and Born effective charges in the DFT import."""

import numpy as np
import pytest

from diffraction import Crystal
from diffraction.phonons.alignment import align_degenerate_modes, conventional_axes
from diffraction.phonons.dft_importers import expand_born_charges, parse_phonopy_born, parse_phonopy_yaml

pytest.importorskip("spglib")

STO = "tests/data/srtio3/band.yaml"  # Gamma point of input/band_srtio3.yaml; BORN next to it
LNO = "tests/data/LiNbO3_band.yaml"


@pytest.fixture(scope="module")
def sto():
    return Crystal.from_dft("STO", STO)


def _gamma(path):
    data = parse_phonopy_yaml(path)
    bands = data["qpoints"][0]["bands"]
    return data, [b["frequency"] for b in bands], np.array([b["eigenvector"] for b in bands])


@pytest.mark.parametrize("path,n_sets,criterion", [(STO, 5, "atoms"), (LNO, 9, "dipole")])
def test_alignment_preserves_subspaces(path, n_sets, criterion):
    data, freqs, E = _gamma(path)
    axes = conventional_axes(data["lattice"], data["species"], data["positions"])
    aligned, labels, weights, sets = align_degenerate_modes(freqs, E, axes, data["masses"])
    optical = [s for s in sets if s["criterion"] == criterion]
    assert len(optical) >= n_sets
    for s in sets:
        b = s["bands"]
        V0, V1 = E[b].reshape(len(b), -1), aligned[b].reshape(len(b), -1)
        np.testing.assert_allclose(V0.T @ V0.conj(), V1.T @ V1.conj(), atol=1e-10)  # same subspace
        np.testing.assert_allclose(V1.conj() @ V1.T, np.eye(len(b)), atol=1e-10)  # orthonormal
        assert min(weights[i] for i in b) > 0.9999


def test_srtio3_soft_mode_partners(sto):
    df = sto.phonons.last_import
    assert list(df["polarization"][:3]) == ["x", "y", "z"]
    for b, axis in enumerate("xyz"):
        u = sto.phonons.get_mode(f"band_q0_b{b}").displacement_cartesian.real
        k = "xyz".index(axis)
        others = np.delete(u, k, axis=1)
        assert np.abs(others).max() < 1e-10  # every atom moves along the partner's axis
    # Frequencies unchanged by the alignment
    data, freqs, _ = _gamma(STO)
    np.testing.assert_allclose(df["frequency_THz"], freqs)


def test_born_charges(sto):
    born = parse_phonopy_born("tests/data/srtio3/BORN")
    assert born["charges"].shape == (3, 3, 3)
    Z = sto.phonons.born["charges"]
    assert Z.shape == (5, 3, 3) and np.abs(Z.sum(axis=0)).max() < 1e-12  # acoustic sum rule
    # O atoms: -5.66 along the Ti-O bond (z, y, x for the three O)
    np.testing.assert_allclose([np.diag(z) for z in Z[2:]],
                               [[-2, -2, -5.66], [-2, -5.66, -2], [-5.66, -2, -2]], atol=1e-12)
    np.testing.assert_allclose(sto.phonons.born["epsilon"], 6.63 * np.eye(3), atol=1e-12)

    for b, axis in enumerate("xyz"):  # soft-mode partners: mode charge along their axis
        z = sto.phonons.get_mode(f"band_q0_b{b}").born_charge
        k = "xyz".index(axis)
        assert abs(z[k]) > 1.5 and np.abs(np.delete(z, k)).max() < 1e-10
    for b in (3, 4, 5, 9, 10, 11):  # acoustic and silent T2u modes: no dipole
        assert np.linalg.norm(sto.phonons.get_mode(f"band_q0_b{b}").born_charge) < 1e-10


def test_born_expansion_wrong_count():
    data = parse_phonopy_yaml(STO)
    with pytest.raises(ValueError, match="independent"):
        expand_born_charges(np.zeros((2, 3, 3)), data["lattice"], data["species"], data["positions"])


def test_alignment_born_save_load(sto, tmp_path):
    sto.save(str(tmp_path) + "/")
    loaded = Crystal.load("STO", str(tmp_path) + "/")
    np.testing.assert_allclose(loaded.phonons.born["charges"], sto.phonons.born["charges"])
    for b in range(3):
        name = f"band_q0_b{b}"
        assert loaded.phonons.get_mode(name).polarization == sto.phonons.get_mode(name).polarization
        np.testing.assert_allclose(loaded.phonons.get_mode(name).born_charge, sto.phonons.get_mode(name).born_charge)


def test_options_off():
    crystal = Crystal.from_dft("STO", STO, align_degenerate=False, born=None)
    df = crystal.phonons.last_import
    assert df["polarization"].isna().all() and df["Z_mode"].isna().all() and crystal.phonons.born is None


def test_lnbo3_hexagonal_alignment():
    """Mapped (hexagonal) cell: E pairs aligned in the basal plane, A modes untouched."""
    hexagonal = Crystal.from_dft("LNO_hex", LNO, cell="conventional")
    rhombohedral = Crystal.from_dft("LNO_rh", LNO)
    pol_hex = list(hexagonal.phonons.last_import["polarization"])
    pol_rh = list(rhombohedral.phonons.last_import["polarization"])
    assert pol_hex == pol_rh
    assert set(p for p in pol_hex if isinstance(p, str)) == {"x", "y"}
