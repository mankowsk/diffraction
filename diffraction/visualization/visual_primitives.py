"""Visual primitives for robotics-based visualization.

This module provides factory functions to create geometric shapes (spheres,
cuboids, cylinders, cones, arrows) as Robotics Toolbox Robot instances for SWIFT.

Each shape returns an rtb.Robot with a `.q` vector for Swift animation:

    Sphere:   q = [x, y, z]
    Cylinder: q = [x, y, z, phi, theta]
    Cone:     q = [x, y, z, phi, theta]
    Arrow:    q = [x, y, z, phi, theta]
    Cuboid:   q = [x, y, z, roll, pitch, yaw]

Angle convention for Cylinder/Cone/Arrow:
    The object points along local +z.
    phi   = azimuth rotation about z
    theta = polar tilt about y (after phi)

Example
-------
>>> from diffraction.visualization.visual_primitives import sphere, cylinder, cuboid
>>> s = sphere(radius=0.2)
>>> c = cylinder(length=1.0, radius=0.05)
"""

from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm


class VisualizationError(Exception):
    """Base exception for visualization-related errors."""
    pass


Color = Tuple[float, float, float, float]
DEFAULT_COLOR: Color = (1.0, 0.607843137255, 0.0, 1.0)  # orange


def _as_color(color: Optional[Sequence[float]]) -> Color:
    """Normalize color to RGBA tuple."""
    if color is None:
        return DEFAULT_COLOR
    if len(color) == 3:
        return (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if len(color) == 4:
        return (float(color[0]), float(color[1]), float(color[2]), float(color[3]))
    raise ValueError("color must be RGB or RGBA")


def _mesh_path(relative_path: str) -> str:
    """Return absolute path relative to the models subfolder of this module.

    Args:
        relative_path: Path relative to diffraction/visualization/models/.

    Returns:
        Absolute path as a string.
    """
    return str((Path(__file__).resolve().parent / "models" / relative_path).resolve())


# =============================================================================
# Shape Factory Functions
# =============================================================================


def sphere(
    name: str = "sphere",
    radius: float = 0.2,
    color: Optional[Sequence[float]] = None,
) -> rtb.Robot:
    """Create a visual sphere robot.

    Args:
        name: Robot name.
        radius: Sphere radius in meters (default 0.2).
        color: RGB or RGBA tuple/list for the sphere color.

    Returns:
        rtb.Robot with q = [x, y, z].
    """
    color = _as_color(color)
    radius = float(radius)

    base = rtb.Link(name="base")
    x = rtb.Link(rtb.ET.tx(), name="x", parent=base, qlim=(-1000, 1000))
    y = rtb.Link(rtb.ET.ty(), name="y", parent=x, qlim=(-1000, 1000))
    z = rtb.Link(rtb.ET.tz(), name="z", parent=y, qlim=(-1000, 1000))

    geom = sg.Sphere(radius=radius, pose=sm.SE3().A, color=color)
    sphere_link = rtb.Link(name="sphere", parent=z, geometry=[geom])

    robot = rtb.Robot([base, x, y, z, sphere_link], name=name)
    robot.q = np.zeros(3)
    return robot


def cylinder(
    name: str = "cylinder",
    length: float = 1.0,
    radius: float = 0.025,
    color: Optional[Sequence[float]] = None,
) -> rtb.Robot:
    """Create a visual cylinder robot.

    The cylinder points along local +z and uses phi/theta for orientation.

    Args:
        name: Robot name.
        length: Cylinder length along its local +z direction (default 1.0).
        radius: Cylinder radius (default 0.025).
        color: RGB or RGBA tuple/list for the cylinder color.

    Returns:
        rtb.Robot with q = [x, y, z, phi, theta].
    """
    color = _as_color(color)
    length = float(length)
    radius = float(radius)

    base = rtb.Link(name="base")
    x = rtb.Link(rtb.ET.tx(), name="x", parent=base, qlim=(-1000, 1000))
    y = rtb.Link(rtb.ET.ty(), name="y", parent=x, qlim=(-1000, 1000))
    z = rtb.Link(rtb.ET.tz(), name="z", parent=y, qlim=(-1000, 1000))

    phi = rtb.Link(rtb.ET.Rz(), name="phi", parent=z)
    theta = rtb.Link(rtb.ET.Ry(), name="theta", parent=phi)

    geom = sg.Cylinder(
        radius=radius, length=length, pose=sm.SE3.Tz(length / 2.0).A, color=color
    )
    cylinder_link = rtb.Link(name="cylinder", parent=theta, geometry=[geom])

    robot = rtb.Robot([base, x, y, z, phi, theta, cylinder_link], name=name)
    robot.q = np.zeros(5)
    return robot


def cuboid(
    name: str = "cuboid",
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    color: Optional[Sequence[float]] = None,
) -> rtb.Robot:
    """Create a visual cuboid robot.

    The cuboid uses full 6-DOF pose: position + roll/pitch/yaw orientation.

    Args:
        name: Robot name.
        a: Side length along local x-axis (default 1.0).
        b: Side length along local y-axis (default 1.0).
        c: Side length along local z-axis (default 1.0).
        color: RGB or RGBA tuple/list for the cuboid color.

    Returns:
        rtb.Robot with q = [x, y, z, roll, pitch, yaw].
    """
    color = _as_color(color)
    a, b, c = float(a), float(b), float(c)

    base = rtb.Link(name="base")
    x = rtb.Link(rtb.ET.tx(), name="x", parent=base, qlim=(-1000, 1000))
    y = rtb.Link(rtb.ET.ty(), name="y", parent=x, qlim=(-1000, 1000))
    z = rtb.Link(rtb.ET.tz(), name="z", parent=y, qlim=(-1000, 1000))

    roll = rtb.Link(rtb.ET.Rx(), name="roll", parent=z)
    pitch = rtb.Link(rtb.ET.Ry(), name="pitch", parent=roll)
    yaw = rtb.Link(rtb.ET.Rz(), name="yaw", parent=pitch)

    geom = sg.Cuboid(scale=[a, b, c], pose=sm.SE3().A, color=color)
    cuboid_link = rtb.Link(name="cuboid", parent=yaw, geometry=[geom])

    robot = rtb.Robot([base, x, y, z, roll, pitch, yaw, cuboid_link], name=name)
    robot.q = np.zeros(6)
    return robot


def cone(
    name: str = "cone",
    scale: float | Sequence[float] = 1.0,
    color: Optional[Sequence[float]] = None,
    mesh: str = "cone.stl",
) -> rtb.Robot:
    """Create a visual cone robot from a mesh file.

    The cone points along local +z and uses phi/theta for orientation.

    Args:
        name: Robot name.
        scale: Mesh scale factor (scalar or [sx, sy, sz]).
        color: RGB or RGBA tuple/list for the cone color.
        mesh: Filename of the cone mesh (default "cone.stl").

    Returns:
        rtb.Robot with q = [x, y, z, phi, theta].

    Raises:
        ValueError: If scale is not a scalar or 3-element sequence.
    """
    color = _as_color(color)

    if isinstance(scale, (list, tuple)):
        if len(scale) != 3:
            raise ValueError("scale must be a scalar or a 3-vector [sx, sy, sz]")
        scale_vec = [float(v) for v in scale]
    else:
        s = float(scale)
        scale_vec = [s, s, s]

    base = rtb.Link(name="base")
    x = rtb.Link(rtb.ET.tx(), name="x", parent=base, qlim=(-1000, 1000))
    y = rtb.Link(rtb.ET.ty(), name="y", parent=x, qlim=(-1000, 1000))
    z = rtb.Link(rtb.ET.tz(), name="z", parent=y, qlim=(-1000, 1000))

    phi = rtb.Link(rtb.ET.Rz(), name="phi", parent=z)
    theta = rtb.Link(rtb.ET.Ry(), name="theta", parent=phi)

    mesh_file = _mesh_path(mesh)
    # Use identity transform - assume mesh has its base at (0,0,0) and extends along +z
    cone_local_pose = sm.SE3().A

    geom = sg.Mesh(filename=mesh_file, scale=scale_vec, pose=cone_local_pose, color=color)
    cone_link = rtb.Link(name="cone", parent=theta, geometry=[geom])

    robot = rtb.Robot([base, x, y, z, phi, theta, cone_link], name=name)
    robot.q = np.zeros(5)
    return robot


def arrow(
    name: str = "arrow",
    length: float = 1.0,
    shaft_radius: float = 0.025,
    cone_scale: float | Sequence[float] = 0.125,
    color: Optional[Sequence[float]] = None,
    mesh: str = "cone.stl",
) -> rtb.Robot:
    """Create a visual arrow robot (cylinder shaft + cone head).

    The arrow points along local +z and uses phi/theta for orientation.

    Args:
        name: Robot name.
        length: Total arrow length including the cone head (default 1.0).
        shaft_radius: Shaft radius (default 0.025).
        cone_scale: Cone scale factor. A scalar or [sx, sy, sz].
                   Default 0.125 gives a ~5:1 ratio with shaft_radius=0.025,
                   which produces visually pleasing proportions.
        color: RGB/RGBA used for both shaft and cone.
        mesh: Filename of the cone mesh (default "cone.stl").

    Returns:
        rtb.Robot with q = [x, y, z, phi, theta].

    Raises:
        ValueError: If arrow length is smaller than or equal to cone length.

    Example:
        >>> # Default proportions (scale/radius ~ 5)
        >>> arrow = arrow(length=1.0)
        >>> # Custom proportions
        >>> arrow = arrow(length=2.0, shaft_radius=0.03, cone_scale=0.15)
    """
    color = _as_color(color)

    if isinstance(cone_scale, (list, tuple)):
        if len(cone_scale) != 3:
            raise ValueError("cone_scale must be a scalar or a 3-vector [sx, sy, sz]")
        cone_vec = [float(v) for v in cone_scale]
    else:
        s = float(cone_scale)
        cone_vec = [s, s, s]

    cone_length = cone_vec[2]
    shaft_length = length - cone_length
    if shaft_length <= 0:
        raise ValueError(
            f"Arrow length ({length}) must be larger than cone length ({cone_length})."
        )

    base = rtb.Link(name="base")
    x = rtb.Link(rtb.ET.tx(), name="x", parent=base, qlim=(-1000, 1000))
    y = rtb.Link(rtb.ET.ty(), name="y", parent=x, qlim=(-1000, 1000))
    z = rtb.Link(rtb.ET.tz(), name="z", parent=y, qlim=(-1000, 1000))

    phi = rtb.Link(rtb.ET.Rz(), name="phi", parent=z)
    theta = rtb.Link(rtb.ET.Ry(), name="theta", parent=phi)

    # Cylinder shaft from z=0 to z=shaft_length (centered at shaft_length/2)
    shaft_geom = sg.Cylinder(
        radius=shaft_radius, length=shaft_length, pose=sm.SE3.Tz(shaft_length / 2.0).A, color=color
    )

    mesh_file = _mesh_path(mesh)
    # Cone head starts at z=shaft_length (base of cone is at shaft_length)
    # Use identity transform - assume mesh has its base at (0,0,0) and extends along +z
    cone_geom = sg.Mesh(
        filename=mesh_file, scale=cone_vec, pose=sm.SE3.Tz(shaft_length).A, color=color
    )

    arrow_link = rtb.Link(name="arrow", parent=theta, geometry=[shaft_geom, cone_geom])

    robot = rtb.Robot([base, x, y, z, phi, theta, arrow_link], name=name)
    robot.q = np.zeros(5)
    return robot


__all__ = [
    "sphere",
    "cylinder",
    "cuboid",
    "cone",
    "arrow",
    "VisualizationError",
]
