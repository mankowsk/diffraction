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
| `add_phonon(name, vectors)` / `delete_phonon(name)` | Manage phonon displacement modes |
| `calc_structure_factor(hkl, energy)` | Calculate static structure factor |
| `calc_structure_factor_displaced(...)` | Calculate F with atomic displacements |
| `save(basepath)` / `load(name, basepath)` | Save/load crystal as JSON |

## Phonons Module

```python
from diffraction.phonons import (
    load_phonon_from_dft,
    normalize_phonon_mode,
    filter_phonon_by_frequency,
)

# Load phonon from DFT calculation
vectors = np.random.rand(4, 3) * 0.1
phonon_data = load_phonon_from_dft(vectors, atom_names=["Sr", "Ti", "O", "O"])

# Normalize mode
normalized = normalize_phonon_mode(vectors, target_amplitude=1.0)
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
