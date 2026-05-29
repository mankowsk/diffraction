import numpy as np
import pandas as pd
import json
import os
import shutil
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub import calc as dccalc
import xrayutilities as xu

class Crystal:
    def __init__(self, name, a=1, b=1, c=1, alpha=90, beta=90, gamma=90):
        self.name = name
        self.unit_cell = {
            "name": name, "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma
        }
        self.ubcalc = dccalc.UBCalculation("you")
        self.ubcalc.set_lattice(**self.unit_cell)
        self.orientations = []
        self.reflections = []
        self.u_matrix = None
        self.ub_matrix = None
        self.xucalc = None  # xrayutilities Crystal object
        self.xu_lattice = None  # xrayutilities SGLattice object
        self.xu_cif_path = None  # Path to the CIF file
        self.phonons = {}  # Dictionary to store phonon modes
        # Default constraints
        self.constraints = {
            "mu": None, "eta": None, "chi": None, "phi": None,
            "delta": None, "gamma": None, "a_eq_b": None,
            "bin_eq_bout": None, "betain": None, "betaout": None,
            "qaz": None, "naz": None, "alpha": None, "beta": None,
            "bisect": None, "psi": None, "omega": None,
        }

    @classmethod
    def from_cif(cls, name, path_to_cif):
        """Create a Crystal from a CIF file, setting unit cell parameters from xrayutilities."""
        xu_cif = xu.materials.cif.CIFFile(path_to_cif)
        xu_lattice = xu_cif.SGLattice()
        xucalc = xu.materials.Crystal(name, xu_lattice)
        unit_cell = {
            "name": name,
            "a": xucalc.a,
            "b": xucalc.b,
            "c": xucalc.c,
            "alpha": xucalc.alpha,
            "beta": xucalc.beta,
            "gamma": xucalc.gamma,
        }
        crystal = cls(**unit_cell)
        crystal.xucalc = xucalc
        crystal.xu_lattice = xu_lattice
        crystal.xu_cif_path = path_to_cif
        return crystal

    def add_orientation(self, hkl, xyz, tag=None):
        """Add a crystal orientation."""
        self.orientations.append({"hkl": hkl, "xyz": xyz, "tag": tag})
        self.ubcalc.add_orientation(hkl, xyz, tag=tag)

    def add_reflection(self, hkl, position, energy, tag=None):
        """Add a reflection."""
        self.reflections.append({"hkl": hkl, "position": position, "energy": energy, "tag": tag})
        self.ubcalc.add_reflection(hkl, position, energy, tag=tag)

    def set_constraints(self, **kwargs):
        """Update constraints."""
        for k, v in kwargs.items():
            if k in self.constraints:
                self.constraints[k] = v

    def calc_ub(self, idx1=0, idx2=1):
        """Calculate UB matrix using two reflections or orientations."""
        self.ubcalc.calc_ub(idx1 + 1, idx2 + 1)
        self.u_matrix = self.ubcalc.U
        self.ub_matrix = self.ubcalc.UB

    def fit_ub(self, indices=None, refine_lattice=False, refine_umatrix=True):
        """Refine UB matrix using reference reflections."""
        ub, lat = self.ubcalc.fit_ub(
            indices, refine_lattice=refine_lattice, refine_umatrix=refine_umatrix
        )
        self.u_matrix = self.ubcalc.U
        self.ub_matrix = self.ubcalc.UB
        return ub, lat

    def calc_angles(self, h, k, l, energy, pandas=True):
        """Calculate diffractometer angles for a given hkl and energy."""
        constraints = Constraints({"nu" if k == "gamma" else k: v for k, v in self.constraints.items()})
        hklcalc = HklCalculation(self.ubcalc, constraints)
        lam = 12398.419843320025 / energy  # eV to Å
        result = hklcalc.get_position(h, k, l, lam)
        if pandas:
            result = pd.concat(
                [
                    pd.DataFrame.from_dict(
                        {
                            **{
                                "gamma" if k == "nu" else k: v
                                for k, v in tres[0].asdict.items()
                            },
                            **tres[1],
                        },
                        orient="index",
                        columns=[f"sol. {n}"],
                    )
                    for n, tres in enumerate(result)
                ],
                axis=1,
            ).T
        return result

    def calc_hkl(
        self, mu, delta, gamma, eta, chi, phi, energy
    ):
        """calculate (h,k,l) for given diffractometer angles and energy in eV.
        """
        angs = [mu, delta, gamma, eta, chi, phi]
        pos = Position(*angs)
        lam = self.en2lam(energy)
        constraints = Constraints({"nu" if k == "gamma" else k: v for k, v in self.constraints.items()})
        hklcalc = HklCalculation(self.ubcalc, constraints)
        try:
            hkl = hklcalc.get_hkl(pos=pos, wavelength=lam)
        except Exception as e:
            print(str(e))
            return
        return hkl
    
    def add_phonon(self, name, vectors):
        """Add a phonon mode to the dictionary."""
        self.phonons[name] = np.array(vectors)

    def delete_phonon(self, name):
        """Delete a phonon mode from the dictionary."""
        if name in self.phonons:
            del self.phonons[name]

    def calc_structure_factor(self, hkl, energy):
        """Calculate structure factor using xrayutilities."""
        if self.xucalc is None or self.xu_lattice is None:
            raise ValueError("xrayutilities Crystal and SGLattice must be set for structure factor calculation.")
        F = []
        for atompos in self.xucalc.lattice._wbase:
            u = atompos[1][1]
            q = self.xucalc.Q(hkl)
            f = atompos[0].f(np.linalg.norm(q), energy)
            F.append(f * np.exp(-2 * np.pi * 1j * np.dot(hkl, u)))
        return np.sum(np.array(F))

    def calc_structure_factor_displaced(self, hkl, energy, phonon_name, amplitude):
        """
        Calculate structure factor for a displaced structure.
        phonon_name: Key of the phonon mode in the dictionary.
        amplitude: Amplitude of the displacement (scalar or array-like).
        """
        if phonon_name not in self.phonons:
            raise ValueError(f"Phonon mode '{phonon_name}' not found.")
        if self.xucalc is None or self.xu_lattice is None:
            raise ValueError("xrayutilities Crystal and SGLattice must be set for structure factor calculation.")

        # Convert amplitude to a numpy array for vectorized operations
        amplitude = np.asarray(amplitude)
        q = self.xucalc.Q(hkl)

        F = []
        for i, atompos in enumerate(self.xucalc.lattice._wbase):
            u = atompos[1][1]
            # Displace atom position by phonon mode and amplitude
            # Use broadcasting to handle both scalar and array amplitudes
            displaced_u = u + amplitude[:, np.newaxis] * self.phonons[phonon_name][i]
            f = atompos[0].f(np.linalg.norm(q), energy)
            # Use np.exp and np.dot with broadcasting
            F.append(f * np.exp(-2 * np.pi * 1j * np.dot(hkl, displaced_u.T)).T)

        return np.sum(np.array(F), axis=0)


            
    def en2lam(self, en):
        """input: energy in eV, returns wavelength in A"""
        return 12398.419843320025 / en

    def lam2en(self, lam):
        """input: wavelength in A, returns energy in eV"""
        return 12398.419843320025 / lam

    def save(self, basepath = "diffraction/data/"):
        path = basepath + self.name
        self.save_to_disk(path)

    def save_to_disk(self, path):
        """Save crystal data to disk, including CIF file and phonons if set."""
        xu_cif_path = os.path.splitext(path)[0] + ".cif"
        json_path = os.path.splitext(path)[0] + ".json"

        data = {
            "name": self.name,
            "unit_cell": self.unit_cell,
            "constraints": self.constraints,
            "orientations": self.orientations,
            "reflections": self.reflections,
            "u_matrix": self.u_matrix.tolist() if self.u_matrix is not None else None,
            "ub_matrix": self.ub_matrix.tolist() if self.ub_matrix is not None else None,
            "xu_cif_path": xu_cif_path,
            "phonons": {k: v.tolist() for k, v in self.phonons.items()},
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        if self.xucalc is not None and self.xu_cif_path is not None:
            # Move CIF file if present but not in same folder
            if not xu_cif_path == self.xu_cif_path:
                shutil.copy2(self.xu_cif_path, xu_cif_path)
                self.xu_cif_path = xu_cif_path
            # Save CIF file if xucalc is set and path is not already set
        elif self.xucalc is not None and self.xu_cif_path is None:
            self.xu_cif_path = xu_cif_path
            self.xucalc.toCIF(self.xu_cif_path)

    @classmethod
    def load(cls, name=None, basepath = "diffraction/data/"):
        if name:
            path = basepath + name + ".json"
        else:
            names = [n.split(".json")[0] for n in os.listdir(basepath) if ".json" in n and not n==".json"]
            names.sort()
        return cls.load_from_disk(path)

    @classmethod
    def load_from_disk(cls, path):
        """Load crystal data from disk, including CIF file and phonons if set."""
        with open(path, "r") as f:
            data = json.load(f)
        crystal = cls(**data["unit_cell"])
        crystal.constraints = data["constraints"]
        for ori in data["orientations"]:
            crystal.add_orientation(ori["hkl"], ori["xyz"], ori.get("tag"))
        for refl in data["reflections"]:
            crystal.add_reflection(refl["hkl"], Position(*refl["position"]), refl["energy"], refl.get("tag"))
        if data["u_matrix"] is not None:
            crystal.u_matrix = np.array(data["u_matrix"])
            crystal.ubcalc.U = crystal.u_matrix
        if data["ub_matrix"] is not None:
            crystal.ub_matrix = np.array(data["ub_matrix"])
            crystal.ubcalc.UB = crystal.ub_matrix
        if data["xu_cif_path"] is not None and os.path.exists(data["xu_cif_path"]):
            xu_cif = xu.materials.cif.CIFFile(data["xu_cif_path"])
            xu_lattice = xu_cif.SGLattice()
            xucalc = xu.materials.Crystal(data["unit_cell"]["name"], xu_lattice)
            crystal.xucalc = xucalc
            crystal.xu_lattice = xu_lattice
            crystal.xu_cif_path = data["xu_cif_path"]
        if "phonons" in data:
            crystal.phonons = {k: np.array(v) for k, v in data["phonons"].items()}
        return crystal
    
    def __repr__(self):
        s= f"### UNIT CELL ###\n"
        for k, v in self.unit_cell.items():
            s+= f"{k:6}: {v}\n"
        s+= f"\n### ORIENTATIONS ###\n"
        s+= f"{'hkl':15}{'xyz':15}{'tag'}\n"
        for orientation in self.orientations:
            s+= f"{str(orientation['hkl']):15}{str(orientation['xyz']):15}{orientation['tag']}\n"
        s+= f"\n### CONSTRAINTS ###\n"
        for k, v in self.constraints.items():
            if not v is None:
                s+= f"{k:6}: {v}\n"
        return s
