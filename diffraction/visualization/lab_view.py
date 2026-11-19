"""Lab-frame view of the diffraction condition with a schematic 4S+2D diffractometer.

Shows the diffractometer (``diffractometer.py``), the sample slab, the crystal axes
(reciprocal a*, b*, c* by default, real a, b, c optional), the incident and scattered beams and Q for a set of diffractometer angles. Geometry comes from
:func:`diffraction.core.lab_geometry.calc_lab_geometry` (diffcalc frame) and is converted to
the SWIFT frame only here, for rendering.

Colors: beams yellow (k_i) / orange (k_f), Q magenta, reciprocal axes a*, b*, c* red / green /
blue, real axes a, b, c light red / light green / light blue (thinner when both are shown),
SwissFEL axes triad x, y, z dark red / dark green / dark blue.
"""

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from diffraction.core.frames import convert
from diffraction.core.lab_geometry import LabGeometry, calc_lab_geometry, to_position
from diffraction.visualization.diffractometer import DEFAULT_DIMENSIONS, JOINTS, load_diffractometer, sample_height
from diffraction.visualization.scene import SwiftScene

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
    "frame_x": (0.55, 0.0, 0.0, 1.0),
    "frame_y": (0.0, 0.45, 0.0, 1.0),
    "frame_z": (0.0, 0.0, 0.6, 1.0),
}


def _rotation_z_to(v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking the local +z axis to the direction of ``v``."""
    z = np.asarray(v, dtype=float) / np.linalg.norm(v)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _pose(R: np.ndarray, t: np.ndarray):
    import spatialmath as sm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return sm.SE3(T, check=False)


class LabVisualization(SwiftScene):
    """Diffraction geometry in the lab: diffractometer, sample, beams and Q.

    Attributes:
        crystal: Crystal with a UB matrix
        energy: X-ray energy in eV
        geometry: :class:`LabGeometry` of the current angles (diffcalc frame)
        origin: Sample position in the SWIFT scene
        robot: Diffractometer robot (q in radians, ordered as ``diffractometer.JOINTS``)

    Example:
        >>> vis = crystal.show_lab(crystal.calc_angles((0, 0, 2), 9000).iloc[0], 9000)
        >>> vis.set_angles({"eta": 12.0, "delta": 24.0})
    """

    def __init__(
        self,
        crystal: "Crystal",
        energy: float,
        beam_length: float = 0.7,
        q_length: float = 0.35,
        axis_length: float = 0.12,
        sample_size: float = 0.08,
        axes: str = "reciprocal",
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
            **dimensions: Diffractometer dimensions, see ``diffractometer.DEFAULT_DIMENSIONS``
        """
        super().__init__()
        self.crystal = crystal
        self.energy = energy
        self.beam_length = beam_length
        self.q_length = q_length
        self.axis_length = axis_length
        self.sample_size = sample_size
        if axes not in ("reciprocal", "real", "both"):
            raise ValueError(f"axes must be 'reciprocal', 'real' or 'both', got '{axes}'")
        self.axes = axes
        self.dimensions = {**DEFAULT_DIMENSIONS, **dimensions}
        self.origin = np.array([0.0, 0.0, sample_height(**dimensions)])
        self.default_look_at = self.origin
        self.robot = load_diffractometer(**dimensions)
        self.geometry: LabGeometry | None = None
        self.shapes: dict[str, Any] = {}
        self._angles: dict[str, float] = {k: 0.0 for k in JOINTS}
        self._create_shapes()

    # SWIFT cannot change the geometry (e.g. arrow length) of a shape already in the scene
    # (no "shape_update" handler in the front end; it would block), only its pose, and
    # env.remove() breaks env.step(). Lengths are therefore set before adding; Q, whose
    # length varies, uses a pool of arrows with lengths rounded to Q_LENGTH_STEP; unused
    # ones are parked below the floor.
    Q_LENGTH_STEP = 0.005
    _HIDDEN = np.array([0.0, 0.0, -10.0])

    @staticmethod
    def _arrow(color, radius: float, length: float):
        import spatialgeometry as sg

        return sg.Arrow(length=length, radius=radius, head_length=min(0.5, 0.03 / length), head_radius=0.5, color=color)

    def _create_shapes(self) -> None:
        import spatialgeometry as sg

        self.shapes["k_i"] = self._arrow(COLORS["k_i"], 0.006, self.beam_length)
        self.shapes["k_f"] = self._arrow(COLORS["k_f"], 0.006, self.dimensions["detector_distance"])
        self._q_pool: dict[float, Any] = {}
        self._q_length: float | None = None
        for kind, names in AXES.items():
            if self.axes in (kind, "both"):
                radius = 0.0025 if (kind == "real" and self.axes == "both") else 0.004
                for ax in names:
                    self.shapes[ax] = self._arrow(COLORS[ax], radius, self.axis_length)
        thickness = self.sample_size / 10
        self.shapes["sample"] = sg.Cuboid(
            scale=[self.sample_size, self.sample_size, thickness], color=COLORS["sample"]
        )

        # SwissFEL axes triad, placed upstream-left on the floor, fixed
        triad_origin = np.array([-0.6, 0.6, 0.02])
        for name in ("x", "y", "z"):
            direction = convert(np.eye(3)["xyz".index(name)], "swissfel", "swift")
            shape = self._arrow(COLORS[f"frame_{name}"], 0.008, 0.25)
            shape.T = _pose(_rotation_z_to(direction), triad_origin)
            self.shapes[f"frame_{name}"] = shape

    def _set_q_arrow(self, length: float, pose) -> None:
        """Show the Q arrow of (rounded) ``length`` at ``pose``; hide the previous one."""
        length = max(self.Q_LENGTH_STEP, round(length / self.Q_LENGTH_STEP) * self.Q_LENGTH_STEP)
        length = round(length, 6)
        if self._q_length is not None and self._q_length != length:
            self._q_pool[self._q_length].T = _pose(np.eye(3), self._HIDDEN)
        if length not in self._q_pool:
            arrow = self._arrow(COLORS["Q"], 0.006, length)
            arrow.T = pose
            self._q_pool[length] = arrow
            if self.env is not None:
                self.env.add(arrow)
        self._q_pool[length].T = pose
        self._q_length = length
        self.shapes["Q"] = self._q_pool[length]

    #: Default camera relative to the sample (SWIFT frame): upstream and above the beam,
    #: looking downstream onto the sample
    CAMERA_POSITION = np.array([-1.0, 0.0, 0.2])
    CAMERA_LOOK_AT = np.array([0.0, 0.0, -0.05])

    def set_camera(self, position: Any = None, look_at: Any = None) -> None:
        """Set the camera; without arguments the default view (upstream, above the beam).

        Args:
            position: Camera position in scene coordinates (default: sample + ``CAMERA_POSITION``)
            look_at: Point the camera looks at (default: sample + ``CAMERA_LOOK_AT``)
        """
        if position is None:
            position = self.origin + self.CAMERA_POSITION
        if look_at is None:
            look_at = self.origin + self.CAMERA_LOOK_AT
        self.env.set_camera_pose(list(position), list(look_at))

    def show(self, angles: Any = None, browser: str | None = None) -> "LabVisualization":
        """Launch SWIFT and draw the scene.

        Args:
            angles: Initial angles (dict, ``calc_angles`` row or diffcalc ``Position``);
                default all zero
            browser: As in :meth:`SwiftScene._launch`; ``"headless"`` for snapshots

        Returns:
            LabVisualization: self
        """
        if self.env is None:
            self._launch(browser)
            self.env.add(self.robot)
            for name, shape in self.shapes.items():
                if name != "Q":
                    self.env.add(shape)
            for arrow in self._q_pool.values():
                self.env.add(arrow)
            self.set_camera()
        self.set_angles(angles if angles is not None else {})
        return self

    @property
    def angles(self) -> dict[str, float]:
        """Current diffractometer angles in degrees (``gamma`` = diffcalc ``nu``)."""
        return {("gamma" if k == "nu" else k): v for k, v in self._angles.items()}

    def set_angles(self, angles: Any = None, energy: float | None = None, **kwargs: float) -> LabGeometry:
        """Move the diffractometer and update sample, beams and Q.

        Angles not given keep their current value.

        Args:
            angles: dict, ``calc_angles`` row or diffcalc ``Position`` (degrees)
            energy: New X-ray energy in eV (optional)
            **kwargs: Individual angles, e.g. ``eta=10`` (``gamma`` or ``nu``)

        Returns:
            LabGeometry: Geometry at the new angles (diffcalc frame)
        """
        if energy is not None:
            self.energy = energy
        given: dict[str, float] = {}
        for source in (angles, kwargs):
            if source is None:
                continue
            position = to_position(source)
            names = source.keys() if isinstance(source, Mapping) or hasattr(source, "keys") else JOINTS
            for name in names:
                key = "nu" if name == "gamma" else name
                if key in JOINTS:
                    given[key] = getattr(position, key)
        self._angles.update(given)

        self.geometry = calc_lab_geometry(self.crystal, self._angles, self.energy)
        self._update_shapes()
        if self.env is not None:
            self.env.step(0)
        return self.geometry

    def _update_shapes(self) -> None:
        geo = self.geometry
        sw = geo.in_frame("swift")
        o = self.origin
        self.robot.q = np.radians([self._angles[j] for j in JOINTS])

        k = np.linalg.norm(sw["k_i"])
        k_i_dir = sw["k_i"] / k
        self.shapes["k_i"].T = _pose(_rotation_z_to(k_i_dir), o - self.beam_length * k_i_dir)
        self.shapes["k_f"].T = _pose(_rotation_z_to(sw["k_f"]), o)

        q_norm = np.linalg.norm(sw["Q"])
        if q_norm / k > 1e-6:
            self._set_q_arrow(round(self.q_length * q_norm / k, 4), _pose(_rotation_z_to(sw["Q"]), o))
        else:  # straight-through beam: tiny Q hidden inside the sample
            self._set_q_arrow(1e-3, _pose(np.eye(3), o))

        for kind, key in (("reciprocal", "reciprocal_lattice"), ("real", "lattice")):
            if self.axes in (kind, "both"):
                for i, ax in enumerate(AXES[kind]):
                    self.shapes[ax].T = _pose(_rotation_z_to(sw[key][:, i]), o)

        # Slab: local z along the surface normal, in-plane edges fixed to the sample
        n_phi = convert(np.asarray(self.crystal.ubcalc.surf_nphi, dtype=float).ravel(), "diffcalc", "swift")
        R_slab = sw["R_sample"] @ _rotation_z_to(n_phi)
        thickness = self.sample_size / 10
        self.shapes["sample"].T = _pose(R_slab, o - R_slab[:, 2] * thickness / 2)

    def summary(self) -> str:
        """Text summary of the current geometry (angles, hkl, 2theta, |Q|)."""
        geo = self.geometry
        ang = ", ".join(f"{k}={v:.3f}" for k, v in self.angles.items())
        hkl = ", ".join(f"{v:.3f}" for v in geo.hkl)
        return (f"{ang}\nE={self.energy:.1f} eV, lambda={geo.wavelength:.4f} A, hkl=({hkl}), "
                f"2theta={geo.two_theta:.3f} deg, |Q|={np.linalg.norm(geo.Q):.4f} 1/A")
