# diffraction

A Python library for X-ray diffraction calculations on crystals and phonon modes.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from diffraction import Crystal

# Load a crystal from CIF file
crystal = Crystal.from_cif("SrTiO3", "diffraction/data/SrTiO3.cif")

# Set orientation relative to laboratory space and calc UB matrix
crystal.add_orientation((0,0,1), (0,0,1), "normal")
crystal.add_orientation((1,0,0), (0,1,0), "inplane along beam")
crystal.calc_ub()

# Set angular constraints
crystal.set_constraints(delta=0, phi=0, eta=0)

# Calculate diffractometer angles for a reflection
angles = crystal.calc_angles(0, 0, 1, 8048.0)
print(angles)

# Calculate structure factor
F = crystal.calc_structure_factor((0, 0, 1), 8048.0)
```

## Package Structure

```
diffraction/
├── core/                    # Core physics/engineering logic
│   ├── crystal.py          # Crystal class (main API)
│   ├── ub_matrix.py        # UB matrix calculation utilities
│   └── reciprocal_lattice.py# Reciprocal space geometry
├── io/                      # Input/output operations
│   ├── json_io.py          # JSON serialization
│   └── cif_parser.py       # CIF file parsing (future)
├── phonons/                 # Phonon/displacement functionality
│   ├── modes.py            # Phonon mode management
│   ├── dft_importers.py    # DFT code importers
│   └── normalization.py    # Mode normalization/filtering
├── visualization/           # Plotting and analysis tools
│   ├── diffraction_pattern.py
│   ├── reciprocal_lattice.py
│   ├── phonon_mode.py
│   └── ub_refinement.py
└── data/                    # Sample CIF/JSON files
```

## Core API: Crystal Class

| Method | Description |
|--------|-------------|
| `Crystal.from_cif(name, path)` | Load crystal from CIF file |
| `calc_angles(h, k, l, energy)` | Get diffractometer angles for a reflection |
| `calc_hkl(mu, delta, gamma, eta, chi, phi, energy)` | Find hkl from measured angles |
| `fit_ub(indices, refine_lattice)` | Refine UB matrix using reference reflections |
| `add_phonon(name, vectors, **kwargs)` / `delete_phonon(name)` | Manage phonon modes |
| `calc_structure_factor(hkl, energy)` | Calculate static structure factor |
| `calc_structure_factor_displaced(...)` | Calculate F with atomic displacements |
| `save(basepath)` / `load(name, basepath)` | Save/load crystal as JSON |

## Phonons Module

```python
from diffraction import Crystal

crystal = Crystal.from_cif("LiNbO3", "diffraction/data/LiNbO3.cif")

# Import all Gamma-point modes from a phonopy calculation (band.yaml with eigenvectors)
summary = crystal.phonons.add_from_dft("band.yaml")
print(summary)  # frequency, acoustic flag and center-of-mass check per mode

# Add a user-defined mode; declare the convention of the input vectors
crystal.phonons.add_mode("soft_mode", vectors, coords="reduced", kind="displacement")

# Modes are stored as mass-normalized displacements (sum m|u|^2 = 1) in reduced
# coordinates; amplitudes are normal coordinates Q in Angstrom * sqrt(amu)
mode = crystal.phonons.get_mode("soft_mode")
Q = mode.amplitude_for_max_displacement(0.1)  # largest atom moves 0.1 Angstrom
F = crystal.calc_structure_factor_displaced((1, 0, 0), 9000, "soft_mode", Q)
```

## Visualization Module

```python
from diffraction.visualization import (
    plot_diffraction_pattern,
    plot_reciprocal_lattice,
    plot_angular_solutions,
)
```

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- diffcalc >= 1.0
- xrayutilities >= 1.6

## Development

Run tests:

```bash
pytest
```

Format code:

```bash
black diffraction/
```

Type check:

```bash
mypy diffraction/
```
