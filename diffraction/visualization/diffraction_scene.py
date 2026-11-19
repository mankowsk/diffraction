"""Shared base for SWIFT views driven by diffractometer angles (lab and reciprocal space).

SWIFT limitations handled here: the geometry (size) of a shape cannot change after it was
added to the scene (no "shape_update" handler in the front end; it would block), only its
pose, and ``env.remove()`` breaks ``env.step()``. Shapes whose size depends on the angles or
the energy therefore come from a :class:`ShapePool` (one shape per rounded size; unused ones
are parked below the floor).
"""

from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from diffraction.core.frames import convert
from diffraction.core.lab_geometry import LabGeometry, calc_lab_geometry, to_position
from diffraction.visualization.diffractometer import JOINTS
from diffraction.visualization.scene import SwiftScene

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

HIDDEN = np.array([0.0, 0.0, -10.0])  # below the SWIFT floor (z = 0), invisible

FRAME_COLORS = {
    "x": (0.55, 0.0, 0.0, 1.0),
    "y": (0.0, 0.45, 0.0, 1.0),
    "z": (0.0, 0.0, 0.6, 1.0),
}


def rotation_z_to(v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking the local +z axis to the direction of ``v``."""
    z = np.asarray(v, dtype=float) / np.linalg.norm(v)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def pose(R: np.ndarray, t: np.ndarray):
    """SE3 pose from rotation ``R`` and translation ``t``."""
    import spatialmath as sm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return sm.SE3(T, check=False)


def arrow(color, radius: float, length: float, head: float = 0.03):
    """``sg.Arrow`` along local +z from the origin, with a head of about ``head`` length."""
    import spatialgeometry as sg

    return sg.Arrow(length=length, radius=radius, head_length=min(0.5, head / length), head_radius=0.5, color=color)


def arrow_pose(start: np.ndarray, vector: np.ndarray):
    """Pose of an arrow from ``start`` along ``vector``."""
    return pose(rotation_z_to(vector), start)


class ShapePool:
    """Shapes of variable size, of which one is visible at a time.

    Args:
        factory: Creates a shape of a given size
        step: Sizes are rounded to multiples of ``step``
    """

    def __init__(self, factory: Callable[[float], Any], step: float) -> None:
        self.factory = factory
        self.step = step
        self.shapes: dict[float, Any] = {}
        self.current: float | None = None
        self.env = None

    def attach(self, env) -> None:
        """Add the shapes created so far to ``env``; later ones are added on creation."""
        self.env = env
        for shape in self.shapes.values():
            env.add(shape)

    def show(self, size: float, shape_pose) -> Any:
        """Show the shape of (rounded) ``size`` at ``shape_pose`` and hide the previous one."""
        size = round(max(self.step, round(size / self.step) * self.step), 6)
        if self.current is not None and self.current != size:
            self.shapes[self.current].T = pose(np.eye(3), HIDDEN)
        if size not in self.shapes:
            shape = self.factory(size)
            shape.T = shape_pose
            self.shapes[size] = shape
            if self.env is not None:
                self.env.add(shape)
        self.shapes[size].T = shape_pose
        self.current = size
        return self.shapes[size]


class DiffractionScene(SwiftScene):
    """Base for views of the diffraction geometry at a set of diffractometer angles.

    Subclasses create their shapes, add them in :meth:`_add_to_env` and update their poses
    in :meth:`_update_shapes` from ``self.geometry`` (a :class:`LabGeometry`).

    Attributes:
        crystal: Crystal with a UB matrix
        energy: X-ray energy in eV
        geometry: :class:`LabGeometry` at the current angles (diffcalc frame)
        origin: Reference point in the SWIFT scene (sample / origin of reciprocal space)
    """

    #: Default camera relative to ``origin`` (SWIFT frame)
    CAMERA_POSITION = np.array([-1.0, 0.0, 0.2])
    CAMERA_LOOK_AT = np.array([0.0, 0.0, -0.05])

    def __init__(self, crystal: "Crystal", energy: float, origin: np.ndarray) -> None:
        super().__init__()
        self.crystal = crystal
        self.energy = energy
        self.origin = np.asarray(origin, dtype=float)
        self.default_look_at = self.origin
        self.geometry: LabGeometry | None = None
        self.shapes: dict[str, Any] = {}
        self._angles: dict[str, float] = {k: 0.0 for k in JOINTS}

    def _add_frame_triad(self, position: np.ndarray, length: float = 0.25, radius: float = 0.008) -> None:
        """Fixed SwissFEL axes triad (x, y, z dark red / green / blue) at ``position``."""
        for name in ("x", "y", "z"):
            direction = convert(np.eye(3)["xyz".index(name)], "swissfel", "swift")
            shape = arrow(FRAME_COLORS[name], radius, length)
            shape.T = arrow_pose(np.asarray(position, dtype=float), direction)
            self.shapes[f"frame_{name}"] = shape

    def _add_to_env(self) -> None:
        """Add all shapes to ``self.env`` (called once after launching)."""
        for shape in self.shapes.values():
            self.env.add(shape)

    def _update_shapes(self) -> None:
        raise NotImplementedError

    def set_camera(self, position: Any = None, look_at: Any = None) -> None:
        """Set the camera; without arguments the default view of this scene.

        Args:
            position: Camera position in scene coordinates (default: origin + ``CAMERA_POSITION``)
            look_at: Point the camera looks at (default: origin + ``CAMERA_LOOK_AT``)
        """
        if position is None:
            position = self.origin + self.CAMERA_POSITION
        if look_at is None:
            look_at = self.origin + self.CAMERA_LOOK_AT
        self.env.set_camera_pose(list(position), list(look_at))

    def show(self, angles: Any = None, browser: str | None = None):
        """Launch SWIFT and draw the scene.

        Args:
            angles: Initial angles (dict, ``calc_angles`` row or diffcalc ``Position``);
                default all zero
            browser: As in :meth:`SwiftScene._launch`; ``"headless"`` for snapshots

        Returns:
            self
        """
        if self.env is None:
            self._launch(browser)
            self._add_to_env()
            self.set_camera()
        self.set_angles(angles if angles is not None else {})
        return self

    @property
    def angles(self) -> dict[str, float]:
        """Current diffractometer angles in degrees (``gamma`` = diffcalc ``nu``)."""
        return {("gamma" if k == "nu" else k): v for k, v in self._angles.items()}

    def set_angles(self, angles: Any = None, energy: float | None = None, **kwargs: float) -> LabGeometry:
        """Set diffractometer angles (and energy) and update the scene.

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

    def summary(self) -> str:
        """Text summary of the current geometry (angles, hkl, 2theta, |Q|).

        If Q ends on an integer hkl, its |F|^2 is given and a systematic absence is flagged.
        """
        geo = self.geometry
        ang = ", ".join(f"{k}={v:.3f}" for k, v in self.angles.items())
        hkl = ", ".join(f"{v:.3f}" for v in geo.hkl)
        text = (f"{ang}\nE={self.energy:.1f} eV, lambda={geo.wavelength:.4f} A, hkl=({hkl}), "
                f"2theta={geo.two_theta:.3f} deg, |Q|={np.linalg.norm(geo.Q):.4f} 1/A")
        target = np.round(geo.hkl).astype(int)
        if np.allclose(geo.hkl, target, atol=1e-3) and target.any():
            F = self.crystal.structure_factor_or_none(tuple(target), self.energy)
            if F is not None:
                text += f"\nQ at ({' '.join(map(str, target))}): |F|^2={abs(F) ** 2:.4g}"
                if self.crystal.is_systematically_absent(tuple(target), self.energy):
                    text += "  -> SYSTEMATICALLY ABSENT, not observable"
        return text
