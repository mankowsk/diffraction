"""Reciprocal lattice points, vectorized structure factors and Ewald excitation errors."""

import numpy as np
import pytest

from diffraction import Crystal
from diffraction.core.reciprocal_space import reciprocal_lattice_points

BI2SE3_CIF = "tests/data/Bi2Se3.cif"


@pytest.fixture
def bi2se3_side():
    """Orientation with the c axis horizontal (normal to the side), as in the user's example."""
    crystal = Crystal.from_cif("Bi2Se3", BI2SE3_CIF)
    crystal.add_orientation((0, 0, 1), (1, 0, 0), "normal to side")
    crystal.add_orientation((1, 0, 0), (0, 0, 1), "a* along beam")
    crystal.calc_ub()
    crystal.set_surface_normal((0, 0, 1))
    crystal.set_constraints(mu=0, chi=90, phi=60)
    return crystal


@pytest.fixture
def bi2se3():
    crystal = Crystal.from_cif("Bi2Se3", BI2SE3_CIF)
    crystal.add_orientation((0, 0, 1), (0, 1, 0), "normal up")
    crystal.add_orientation((1, 0, 0), (0, 0, 1), "a* along beam")
    crystal.calc_ub()
    crystal.set_constraints(mu=0, chi=0, phi=0)
    return crystal


def test_structure_factors_vectorized(bi2se3):
    hkls = [(2, 0, 20), (0, 0, 3), (1, 0, 1), (0, 0, 1), (1, 1, 0)]
    F = bi2se3.calc_structure_factors(hkls, 7500)
    np.testing.assert_allclose(F, [bi2se3.calc_structure_factor(h, 7500) for h in hkls], atol=1e-9)
    F_plain = bi2se3.calc_structure_factors(hkls, 7500, occupancy=False, debye_waller=False)
    np.testing.assert_allclose(F_plain, [bi2se3.calc_structure_factor(h, 7500, False, False) for h in hkls], atol=1e-9)


def test_reciprocal_lattice_points(bi2se3):
    energy = 7500
    k = 2 * np.pi / bi2se3.en2lam(energy)
    df = reciprocal_lattice_points(bi2se3, energy, max_points=None)
    assert df["q"].max() <= 2 * k + 1e-9
    assert not ((df["h"] == 0) & (df["k"] == 0) & (df["l"] == 0)).any()
    # R-centring (-h + k + l = 3n) removes all other reflections as systematic absences
    assert ((-df["h"] + df["k"] + df["l"]) % 3 == 0).all()
    # Completeness: all allowed reflections inside the limiting sphere are present
    UB = np.asarray(bi2se3.ubcalc.UB)
    present = set(zip(df["h"], df["k"], df["l"]))
    for hkl in [(0, 0, 15), (1, 0, 1), (0, 1, 5), (1, 0, 10)]:
        assert np.linalg.norm(UB @ hkl) <= 2 * k
        assert hkl in present
    capped = reciprocal_lattice_points(bi2se3, energy, max_points=50)
    assert len(capped) == 50 and capped["intensity"].min() >= df["intensity"].nlargest(50).min()


def test_excitation_error_zero_at_solutions(bi2se3):
    energy = 7500
    for hkl in [(0, 0, 15), (1, 0, 10)]:
        for _, row in bi2se3.calc_angles(hkl, energy).iterrows():
            geo = bi2se3.calc_lab_geometry(row, energy)
            assert abs(geo.excitation_error(hkl)[0]) < 1e-9
            # (000) always lies on the Ewald sphere
            assert geo.excitation_error([(0, 0, 0)])[0] == pytest.approx(0, abs=1e-12)


def test_calc_angles_intensity_and_absence_warning(bi2se3_side):
    import warnings

    from diffraction.core.crystal import SystematicAbsenceWarning

    bi2se3 = bi2se3_side
    with pytest.warns(SystematicAbsenceWarning, match=r"\(2, 0, 19\)"):
        df = bi2se3.calc_angles((2, 0, 19), 7500)
    assert (df["F2"] < 1e-12).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SystematicAbsenceWarning)
        df = bi2se3.calc_angles((2, 0, 20), 7500)
        raw = bi2se3.calc_angles((2, 0, 20), 7500, pandas=False)
    expected = abs(bi2se3.calc_structure_factor((2, 0, 20), 7500)) ** 2
    np.testing.assert_allclose(df["F2"].to_numpy(dtype=float), expected)
    assert raw[0][1]["F2"] == pytest.approx(expected)
    assert df["eta"].dtype == float  # rows stay real

    # Without atomic positions: F2 is None, no warning
    plain = Crystal("cubic", a=4, b=4, c=4)
    plain.add_orientation((0, 0, 1), (0, 1, 0), "normal up")
    plain.add_orientation((1, 0, 0), (0, 0, 1), "a along beam")
    plain.calc_ub()
    plain.set_constraints(mu=0, chi=0, phi=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SystematicAbsenceWarning)
        df = plain.calc_angles((0, 0, 2), 9000)
    assert df["F2"].isna().all()
    assert plain.is_systematically_absent((1, 0, 0), 9000) is None


def test_detector_angles(bi2se3_side):
    """delta/gamma predicted for a reflection reproduce its k_f; at a solution they equal the angles."""
    import warnings

    bi2se3 = bi2se3_side
    energy = 7500
    for hkl in [(2, 0, 20), (1, 0, 10), (0, 1, 5)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = bi2se3.calc_angles(hkl, energy)
        for _, row in df.iterrows():
            geo = bi2se3.calc_lab_geometry(row, energy)
            delta, gamma, two_theta = (v[0] for v in geo.detector_angles([hkl]))
            d, g = np.radians(delta), np.radians(gamma)
            k_f = np.array([np.sin(d), np.cos(d) * np.cos(g), np.cos(d) * np.sin(g)]) * np.linalg.norm(geo.k_i)
            np.testing.assert_allclose(k_f, geo.k_f, atol=1e-9)
            assert two_theta == pytest.approx(geo.two_theta, abs=1e-8)
            if abs(row["delta"]) <= 90:
                assert delta == pytest.approx(row["delta"], abs=1e-8)
                assert gamma == pytest.approx(row["gamma"], abs=1e-8)
