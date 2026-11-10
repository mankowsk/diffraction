# diffraction

Tools to plan time-resolved X-ray diffraction experiments and to analyse
structural changes due to phonon-driven distortions at the
[Bernina beamline](https://www.psi.ch/en/swissfel/bernina) of SwissFEL (Paul
Scherrer Institut).

`diffraction` puts everything needed for a single-crystal diffraction
experiment into one `Crystal` object: the crystal structure (from a CIF file),
the sample orientation (UB matrix), the diffractometer constraints and a set of
phonon modes. From these it calculates where a reflection appears on the
diffractometer and how its structure factor changes when the lattice is
distorted along a phonon mode. It provides:

- **Diffractometer angles**: all angle solutions (`mu`, `delta`, `gamma`, `eta`,
  `chi`, `phi`) for a reflection at a given photon energy, under user-defined
  constraints, and the inverse calculation from measured angles back to *hkl*.
- **Sample orientation**: UB matrix from orientation constraints and/or measured
  reflections, with least-squares refinement of the U matrix and, optionally, the
  lattice parameters.
- **Structure factors**: static structure factors F(*hkl*) and structure factors
  of a crystal distorted along a Γ-point phonon mode, as a function of the mode
  amplitude.
- **Phonon modes**: phonon modes imported from phonopy calculations or defined
  manually, stored in a single well-defined convention (mass-normalized
  displacements, normal-coordinate amplitudes), with consistency checks.
- **3D visualization**: interactive unit cell views and phonon mode animations
  in the browser using SWIFT.
- **Persistence**: crystals (orientation, constraints, phonon modes) saved to and
  loaded from JSON.

## Installation

Requires Python ≥ 3.10. The package is not on PyPI, so install it from a clone of
this repository:

```bash
git clone https://github.com/mankowsk/diffraction.git
cd diffraction
pip install -e .
```

This installs the core dependencies (`numpy`, `pandas`, `diffcalc`,
`xrayutilities`), which is enough for everything except the 3D (SWIFT)
visualization.

### Visualization (SWIFT animations)

`Crystal.show_unit_cell()`, `Crystal.show_phonon()` and the contents of
`diffraction.visualization` need `swift-sim` and `roboticstoolbox-python`.
**These must be installed from the following forks, not from PyPI:**

- `swift-sim`: <https://github.com/mankowsk/swift>
- `roboticstoolbox-python`: <https://github.com/PhilFreeman/robotics-toolbox-python>

The released packages are outdated, and fixes are not currently being reviewed
upstream. The `visualization` extra pins both forks:

```bash
pip install -e ".[visualization]"
# or, equivalently
pip install -r requirements-visualization.txt
```

Notes:

- Install both forks **in the same `pip` command** (as above). `roboticstoolbox-python`
  depends on `swift-sim`, so if you install it alone, pip pulls the outdated
  `swift-sim` from PyPI instead of the fork.
- If the PyPI versions are already installed, replace them:
  ```bash
  pip uninstall -y swift-sim roboticstoolbox-python
  pip install -r requirements-visualization.txt
  ```
- Both packages build C extensions from source, so you need a C/C++ compiler
  (on macOS: `xcode-select --install`) and `git`.
- To check which version is installed, run `pip inspect`. Each package's
  `direct_url` should point to the GitHub fork.
- The 2D plots (`plot_angular_solutions`) also need `matplotlib`, which is not
  installed automatically.

### Development

```bash
pip install -e ".[dev]"            # pytest, black, mypy, flake8
pip install -e ".[dev,visualization]"
```

## Quick Start

```python
from diffraction import Crystal

# Load a crystal from a CIF file
crystal = Crystal.from_cif("SrTiO3", "diffraction/data/SrTiO3.cif")

# Orient the crystal in the laboratory frame and calculate the UB matrix
crystal.add_orientation((0, 0, 1), (0, 0, 1), "normal")
crystal.add_orientation((1, 0, 0), (0, 1, 0), "inplane along beam")
crystal.calc_ub()

# Fix three diffractometer angles
crystal.set_constraints(delta=0, phi=0, eta=0)

# All angle solutions for a reflection at 8048 eV (pandas DataFrame)
angles = crystal.calc_angles((0, 0, 1), 8048.0)
print(angles)

# Static structure factor
F = crystal.calc_structure_factor((0, 0, 1), 8048.0)

# Save the crystal (orientation, constraints, phonons) and load it again
crystal.save()                       # -> diffraction/data/SrTiO3.json
crystal = Crystal.load("SrTiO3")
```

The other way round, from measured angles to *hkl*:

```python
hkl = crystal.calc_hkl(mu=0, delta=0, gamma=30.0, eta=0, chi=0, phi=0, energy=8048.0)
```

## Core API: the `Crystal` class

| Method | Description |
|--------|-------------|
| `Crystal.from_cif(name, path)` | Load a crystal from a CIF file |
| `add_orientation(hkl, xyz, tag)` | Add an orientation constraint (*hkl* parallel to lab direction `xyz`) |
| `add_reflection(hkl, position, energy, tag)` | Add a measured reflection |
| `calc_ub(idx1, idx2)` | Calculate U and UB from two orientations/reflections |
| `fit_ub(indices, refine_lattice, refine_umatrix)` | Refine UB (and optionally the lattice) by least squares |
| `set_constraints(**kwargs)` | Set diffractometer constraints (`mu`, `delta`, `gamma`, `eta`, `chi`, `phi`, …) |
| `calc_angles(hkl, energy, pandas=True)` | All angle solutions for a reflection |
| `calc_hkl(mu, delta, gamma, eta, chi, phi, energy)` | *hkl* from measured angles |
| `calc_structure_factor(hkl, energy)` | Static structure factor F(*hkl*) |
| `calc_structure_factor_displaced(hkl, energy, phonon_name, amplitude)` | F for a crystal displaced along a phonon mode (scalar or array of amplitudes) |
| `add_phonon(name, vectors, **kwargs)` / `delete_phonon(name)` | Add or remove a phonon mode (shortcuts for `crystal.phonons`) |
| `en2lam(en)` / `lam2en(lam)` | Photon energy (eV) ↔ wavelength (Å) |
| `show_unit_cell(distance, browser)` / `show_phonon(mode_name, ...)` | 3D view of the unit cell / phonon animation (SWIFT) |
| `save(basepath)` / `Crystal.load(name, basepath)` | Save to / load from JSON (default `diffraction/data/`) |

### Conventions

Diffractometer angles and the UB matrix follow
[diffcalc](https://github.com/DiamondLightSource/diffcalc-core) and the **You
convention** for a "4S+2D" six-circle diffractometer (H. You, *J. Appl. Cryst.*
**32**, 614–623 (1999)):

- sample circles `mu`, `eta`, `chi`, `phi`; detector circles `delta`, `gamma`
  (`gamma` is called `nu` in diffcalc and in You's paper);
- laboratory frame: the incident beam is along +y. At zero angles, `delta`
  moves the detector in the x–y plane and `gamma` in the y–z plane.
  `add_orientation(hkl, xyz)` uses lab directions `xyz` in this frame.

Energies are in eV, lengths in Å and angles in degrees. When a crystal is saved,
its CIF file is copied next to the JSON file and referenced from it, not
embedded.

## Phonon modes

Phonon modes are managed by `crystal.phonons` (a `PhononsManager`). Each mode is
a `PhononMode` and is converted on input to a single convention:

- **mass-normalized displacements** *u* with Σ m |u|² = 1 (masses in amu), so
  mode amplitudes are **normal coordinates Q in Å·√amu**;
- in **reduced coordinates** of the crystal lattice;
- with the global phase fixed so that the largest component is real and
  positive.

The DFT-style eigenvector (e = √m · u, Σ |e|² = 1) is available as
`mode.eigenvector`. `mode.consistency` reports the normalization, whether the
mode is acoustic and whether it conserves the center of mass.

```python
from diffraction import Crystal

crystal = Crystal.from_cif("LiNbO3", "diffraction/data/LiNbO3.cif")

# Import modes from phonopy (band.yaml, qpoints.yaml or mesh.yaml with eigenvectors).
# DFT atoms are matched to the CIF atoms by species and position and reordered.
summary = crystal.phonons.add_from_dft("band.yaml", q_index=0)
print(summary)  # frequency, acoustic flag and center-of-mass check per mode

# Or add a mode by hand; declare the convention of the input vectors
crystal.phonons.add_mode("soft_mode", vectors, coords="reduced", kind="displacement")

# Structure factor for a distortion in which the largest atomic displacement is 0.1 Å
mode = crystal.phonons.get_mode("soft_mode")
Q = mode.amplitude_for_max_displacement(0.1)
F = crystal.calc_structure_factor_displaced((1, 0, 0), 9000, "soft_mode", Q)
```

Current limitations:

- The CIF file must list **all atoms of the unit cell explicitly** (space group
  P1), in the same order as the mode vectors. (`add_from_dft` reorders the
  atoms for you.)
- Structure factors and animations support **Γ-point modes only** (modes at
  q ≠ 0 can be stored but not used for these calculations).
- Only phonopy files can be imported. VASP, Quantum ESPRESSO and ABINIT
  importers are not implemented yet.

## Visualization

3D views run in the browser through SWIFT (see
[Installation](#visualization-swift-animations)):

```python
import numpy as np

crystal.show_unit_cell()
crystal.phonons.show("soft_mode")    # sine oscillation, 0.2 Å max displacement

# Custom time trace of the normal coordinate Q
mode = crystal.phonons.get_mode("soft_mode")
t = np.linspace(0, 20 * np.pi, 500)
crystal.phonons.show("soft_mode",
                     amplitudes=mode.amplitude_for_max_displacement(0.1) * np.sin(t))
```

Atoms are drawn at their true positions and displacements, including periodic
images of atoms on the cell boundary. Basic shapes (`sphere`, `cylinder`,
`cuboid`, `cone`, `arrow`) from `diffraction.visualization` can be added to a
`CrystalVisualization` and animated.

For 2D plots, `plot_angular_solutions(crystal, hkl, energy)` shows the angle
solutions for a reflection. The other `plot_*` functions in
`diffraction.visualization` are placeholders and not implemented yet.

## Examples

Jupyter notebooks in `examples/`:

| Notebook | Content |
|----------|---------|
| `basic_usage.ipynb` | Full workflow: crystal, UB matrix, angles, unit cell view, save/load, animation, constraint sensitivity |
| `visualization.ipynb` | 3D visualization of scattering geometries |
| `visualization_animated.ipynb` | Animating atoms along displacement vectors |
| `visualization_ewald_sphere.ipynb` | Ewald sphere |
| `Bi2Se3.ipynb`, `TTO.ipynb` | The basic workflow applied to Bi2Se3 and Tb2Ti2O7 |

Sample CIF files and saved crystals (SrTiO3, LiNbO3, Sr2IrO4) are in
`diffraction/data/`.

## Package Structure

```
diffraction/
├── core/
│   ├── crystal.py              # Crystal class (main API)
│   ├── ub_matrix.py            # UB matrix utilities
│   └── reciprocal_lattice.py   # Reciprocal space geometry
├── io/
│   └── json_io.py              # JSON serialization of Crystal objects
├── phonons/
│   ├── modes.py                # PhononMode: storage convention and checks
│   ├── manager.py              # PhononsManager (crystal.phonons)
│   ├── dft_importers.py        # phonopy importer (VASP/QE/ABINIT: planned)
│   └── normalization.py        # Legacy helpers on raw arrays
├── visualization/
│   ├── crystal_visualization.py  # CrystalVisualization (SWIFT)
│   ├── visual_primitives.py      # sphere, cylinder, cuboid, cone, arrow
│   ├── angular_solutions.py      # 2D plots of angle solutions
│   └── ...                       # placeholder plotting modules
└── data/                       # Sample CIF and JSON files
```

## Development

```bash
pytest                    # run from the repository root
pytest --cov=diffraction  # with coverage
black diffraction/        # format (line length 88)
mypy diffraction/         # type check
```

## Acknowledgements

`diffraction` relies on the following open-source projects:

- [xrayutilities](https://github.com/dkriegner/xrayutilities): crystal
  structures from CIF files, atomic form factors and structure factors.
  D. Kriegner, E. Wintersberger and J. Stangl, *J. Appl. Cryst.* **46**,
  1162–1170 (2013).
- [diffcalc](https://github.com/DiamondLightSource/diffcalc-core) (Diamond
  Light Source): UB matrix calculation and refinement, and diffractometer
  angles in the You convention.
- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
  and [Swift](https://github.com/jhavl/swift) (P. Corke, J. Haviland): 3D
  visualization and animation in the browser. This project uses the forks
  listed under [Installation](#visualization-swift-animations).

Phonon modes can be imported from [phonopy](https://phonopy.github.io/phonopy/)
output files.

## Author

Roman Mankowsky (roman.mankowsky@psi.ch), Paul Scherrer Institut

## License

MIT License. See [LICENSE](LICENSE).
