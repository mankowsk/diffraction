"""Lab-frame view of the diffraction condition with a schematic 4S+2D diffractometer.

Shows the diffractometer (``diffractometer.py``), the sample slab, the crystal axes
(reciprocal a*, b*, c* by default, real a, b, c optional), the incident beam, the detector
direction and Q_det = k_det - k_i (what the detector is set to measure) and the diffracted
beams k_i + G of all reflections near the Bragg condition, for a set of diffractometer angles. Geometry comes from
:func:`diffraction.core.lab_geometry.calc_lab_geometry` (diffcalc frame) and is converted to
the SWIFT frame only here, for rendering.

Colors: incident beam yellow, detector direction orange (thin), Q_det magenta, diffracted
beams red (thickness by intensity class: rel |F|^2 >= 0.1, >= 0.01, below), detector hits green, reciprocal axes a*, b*, c* red / green /
blue, real axes a, b, c light red / light green / light blue (thinner when both are shown),
SwissFEL axes triad x, y, z dark red / dark green / dark blue.
"""

from typing import TYPE_CHECKING

import numpy as np

from diffraction.core.frames import convert
from diffraction.visualization.diffraction_scene import (
    DIFFRACTED_COLOR,
    JOINTS,
    DiffractionScene,
    ShapeSet,
    intensity_class,
    ShapePool,
    arrow,
    arrow_pose,
    pose,
    rotation_z_to,
)
from diffraction.visualization.diffractometer import DEFAULT_DIMENSIONS, load_diffractometer, sample_height

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

AXES = {"reciprocal": ("a*", "b*", "c*"), "real": ("a", "b", "c")}

COLORS = {
    "k_i": (1.0, 0.9, 0.1, 1.0),
    "k_f": (1.0, 0.55, 0.0, 1.0),
    "Q": (0.85, 0.1, 0.85, 1.0),
    "a*": (0.9, 0.1, 0.1, 1.0),
    "b*": (0.1, 0.75, 0.1, 1.0),
    "c*": (0.15, 0.3, 1.0, 1.0),
    "a": (1.0, 0.6, 0.6, 1.0),
    "b": (0.6, 1.0, 0.6, 1.0),
    "c": (0.6, 0.75, 1.0, 1.0),
    "sample": (0.3, 0.75, 0.95, 1.0),
    "hit": (0.2, 1.0, 0.3, 1.0),
}


class LabVisualization(DiffractionScene):
    """Diffraction geometry in the lab: diffractometer, sample, beams and Q.

    Attributes:
        robot: Diffractometer robot (q in radians, ordered as ``diffractometer.JOINTS``)
        origin: Sample position in the SWIFT scene

    Example:
        >>> vis = crystal.show_lab(crystal.calc_angles((0, 0, 2), 9000).iloc[0], 9000)
        >>> vis.set_angles({"eta": 12.0, "delta": 24.0})
    """

    #: Default camera relative to the sample: upstream and above the beam, looking downstream
    CAMERA_POSITION = np.array([-1.0, 0.0, 0.2])
    CAMERA_LOOK_AT = np.array([0.0, 0.0, -0.05])
    Q_LENGTH_STEP = 0.005
    BEAM_RADII = (0.006, 0.0035, 0.002)  # per intensity class (strong, medium, weak)

    def __init__(
        self,
        crystal: "Crystal",
        energy: float,
        beam_length: float = 0.7,
        q_length: float = 0.35,
        axis_length: float = 0.12,
        sample_size: float = 0.08,
        axes: str = "reciprocal",
        beams: bool = True,
        max_beams: int = 12,
        tolerance: float | None = None,
        q_max: float | None = None,
        min_intensity: float = 1e-3,
        **dimensions: float,
    ) -> None:
        """Set up the scene (call :meth:`show` to render it).

        Args:
            crystal: Crystal with a UB matrix (``calc_ub``)
            energy: X-ray energy in eV
            beam_length: Length of the incident beam drawn upstream of the sample
            q_length: Drawn length of Q for |Q| = |k| (Q is drawn to scale relative to k)
            axis_length: Drawn length of the crystal axis arrows (directions only)
            sample_size: Edge length of the sample slab
            axes: Crystal axes to draw: ``"reciprocal"`` (a*, b*, c*; default, matching the
                hkl of orientations and Q = h a* + k b* + l c*), ``"real"`` (a, b, c) or ``"both"``
            beams: Draw the diffracted beams of reflections near the Bragg condition
            max_beams: Maximum number of diffracted beams drawn (strongest first)
            tolerance: Excitation error (1/Angstrom) for "near the Bragg condition"
                (default 1 % of k, as in the reciprocal view)
            q_max, min_intensity: Reflections considered (see ``reciprocal_lattice_points``)
            **dimensions: Diffractometer dimensions, see ``diffractometer.DEFAULT_DIMENSIONS``
        """
        super().__init__(crystal, energy, origin=np.array([0.0, 0.0, sample_height(**dimensions)]))
        if axes not in ("reciprocal", "real", "both"):
            raise ValueError(f"axes must be 'reciprocal', 'real' or 'both', got '{axes}'")
        self.beam_length = beam_length
        self.q_length = q_length
        self.axis_length = axis_length
        self.sample_size = sample_size
        self.axes = axes
        self.dimensions = {**DEFAULT_DIMENSIONS, **dimensions}
        self.beams = beams
        self.max_beams = max_beams
        self._init_reflections(q_max=q_max, min_intensity=min_intensity, tolerance=tolerance)
        self.robot = load_diffractometer(**dimensions)
        self._create_shapes()

    def _create_shapes(self) -> None:
        import spatialgeometry as sg

        self.shapes["k_i"] = arrow(COLORS["k_i"], 0.006, self.beam_length)
        # Detector direction (thin): where the detector looks, not a physical beam
        self.shapes["k_f"] = arrow(COLORS["k_f"], 0.0025, self.dimensions["detector_distance"])
        # Diffracted beams, ending at the detector face when aimed at the detector
        beam_length = self._detector_face()
        self.beam_sets = [
            ShapeSet(lambda length, r=r: arrow(DIFFRACTED_COLOR, r, length), self.max_beams)
            for r in self.BEAM_RADII
        ]
        self.hit_markers = ShapeSet(lambda r: sg.Sphere(radius=r, color=COLORS["hit"]), self.max_beams)
        self._beam_length = beam_length
        self.q_pool = ShapePool(lambda length: arrow(COLORS["Q"], 0.006, length), self.Q_LENGTH_STEP)
        for kind, names in AXES.items():
            if self.axes in (kind, "both"):
                radius = 0.0025 if (kind == "real" and self.axes == "both") else 0.004
                for ax in names:
                    self.shapes[ax] = arrow(COLORS[ax], radius, self.axis_length)
        thickness = self.sample_size / 10
        self.shapes["sample"] = sg.Cuboid(
            scale=[self.sample_size, self.sample_size, thickness], color=COLORS["sample"]
        )
        self._add_frame_triad(np.array([-0.6, 0.6, 0.02]))

    def _add_to_env(self) -> None:
        self.env.add(self.robot)
        super()._add_to_env()
        self.q_pool.attach(self.env)
        for shape_set in (*self.beam_sets, self.hit_markers):
            shape_set.attach(self.env)

    def _detector_face(self) -> float:
        """Distance of the detector's front face from the sample (along the detector direction)."""
        return self.dimensions["detector_distance"] + 0.005

    def detector_hits(self, beams) -> tuple[np.ndarray, np.ndarray]:
        """Which diffracted beams hit the detector plate, and where (SWIFT scene).

        Args:
            beams: DataFrame from :meth:`in_condition` (columns ``kx, ky, kz``)

        Returns:
            tuple: (bool array, (N, 3) hit points on the detector face)
        """
        directions = convert(beams[["kx", "ky", "kz"]].to_numpy(dtype=float), "diffcalc", "swift")
        R_det = convert(self.geometry.R_detector, "diffcalc", "swift", kind="rotation")
        local = directions @ R_det  # components along the detector's x (normal), y, z axes
        face = self._detector_face()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(local[:, 0] > 1e-9, face / local[:, 0], np.inf)
        half = self.dimensions["detector_size"] / 2
        hit = np.isfinite(t) & (np.abs(local[:, 1] * t) <= half) & (np.abs(local[:, 2] * t) <= half)
        points = self.origin + directions * np.where(np.isfinite(t), t, 0.0)[:, None]
        return hit, points

    def _table_columns(self, hits) -> dict:
        hit, _ = self.detector_hits(hits)
        return {"det": np.where(hit, "HIT", "")}

    def _update_shapes(self) -> None:
        sw = self.geometry.in_frame("swift")
        o = self.origin
        self.robot.q = np.radians([self._angles[j] for j in JOINTS])

        k = np.linalg.norm(sw["k_i"])
        k_i_dir = sw["k_i"] / k
        self.shapes["k_i"].T = arrow_pose(o - self.beam_length * k_i_dir, k_i_dir)
        self.shapes["k_f"].T = arrow_pose(o, sw["k_f"])

        q_norm = np.linalg.norm(sw["Q"])
        if q_norm / k > 1e-6:
            self.q_pool.show(self.q_length * q_norm / k, arrow_pose(o, sw["Q"]))
        else:  # straight-through beam: tiny Q hidden inside the sample
            self.q_pool.show(1e-3, pose(np.eye(3), o))

        for kind, key in (("reciprocal", "reciprocal_lattice"), ("real", "lattice")):
            if self.axes in (kind, "both"):
                for i, ax in enumerate(AXES[kind]):
                    self.shapes[ax].T = arrow_pose(o, sw[key][:, i])

        if self.beams:
            self._update_beams()

        # Slab: local z along the surface normal, in-plane edges fixed to the sample
        n_phi = convert(np.asarray(self.crystal.ubcalc.surf_nphi, dtype=float).ravel(), "diffcalc", "swift")
        R_slab = sw["R_sample"] @ rotation_z_to(n_phi)
        thickness = self.sample_size / 10
        self.shapes["sample"].T = pose(R_slab, o - R_slab[:, 2] * thickness / 2)

    def _update_beams(self) -> None:
        """Diffracted beams of the strongest reflections near the Bragg condition."""
        beams = self.in_condition().nlargest(self.max_beams, "intensity")
        directions = convert(beams[["kx", "ky", "kz"]].to_numpy(dtype=float), "diffcalc", "swift")
        classes = [intensity_class(r) for r in beams["rel_intensity"]]
        for c, beam_set in enumerate(self.beam_sets):
            beam_set.show(self._beam_length,
                          [arrow_pose(self.origin, d) for d, cls in zip(directions, classes) if cls == c])
        hit, points = self.detector_hits(beams)
        self.hit_markers.show(0.012, [pose(np.eye(3), p) for p in points[hit]])
