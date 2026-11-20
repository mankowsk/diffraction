"""Reciprocal-space view: Ewald sphere and reciprocal lattice in the lab frame.

The reciprocal lattice (points sized by |F|^2) rotates with the sample angles; the Ewald
sphere (radius k = 2 pi / lambda) is fixed in the lab, centred at -k_i from the origin of
reciprocal space. Reflections within ``tolerance`` of the sphere are in diffraction
condition and highlighted. Everything is drawn in the lab frame (converted to SWIFT only
here), with ``scale`` scene units per 1/Angstrom.

Colors: lattice points steel blue, (000) white, reflections in diffraction condition
red halo, Ewald sphere yellow wireframe (limiting sphere grey), k_i yellow / k_f orange (from the sphere centre),
Q magenta (from the origin), diffracted beams k_f = k_i + G of the highlighted reflections
red (from the sphere centre; thickness by intensity class), a*, b*, c* red / green / blue, SwissFEL axes triad dark red /
dark green / dark blue.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from diffraction.core.frames import convert
from diffraction.core.reciprocal_space import reciprocal_lattice_points
from diffraction.visualization.diffraction_scene import (
    DIFFRACTED_COLOR,
    HIDDEN,
    DiffractionScene,
    ShapePool,
    ShapeSet,
    arrow,
    arrow_pose,
    intensity_class,
    pose,
)
from diffraction.visualization.meshes import MODELS_DIR, SPHERE_WIREFRAME_FILE

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal

COLORS = {
    "point": (0.35, 0.6, 0.85, 1.0),
    "origin": (1.0, 1.0, 1.0, 1.0),
    "highlight": (1.0, 0.15, 0.1, 0.45),
    "sphere": (0.95, 0.85, 0.3, 1.0),
    "limiting_sphere": (0.6, 0.6, 0.6, 1.0),
    "k_i": (1.0, 0.9, 0.1, 1.0),
    "k_f": (1.0, 0.55, 0.0, 1.0),
    "Q": (0.85, 0.1, 0.85, 1.0),
    "a*": (0.9, 0.1, 0.1, 1.0),
    "b*": (0.1, 0.75, 0.1, 1.0),
    "c*": (0.15, 0.3, 1.0, 1.0),
}


class ReciprocalVisualization(DiffractionScene):
    """Ewald construction in the lab frame for a set of diffractometer angles.

    Attributes:
        points: DataFrame of the drawn reflections (``h, k, l, q, F, intensity,
            rel_intensity``, see :func:`reciprocal_lattice_points`)
        scale: Scene units per 1/Angstrom
        tolerance: Excitation error (1/Angstrom) below which a reflection counts as in
            diffraction condition
        origin: Origin of reciprocal space (000) in the SWIFT scene

    Example:
        >>> vis = crystal.show_reciprocal(crystal.calc_angles((0, 0, 15), 7500).iloc[0], 7500)
        >>> vis.in_condition()
    """

    SIZE_STEP = 0.002  # rounding of variable sizes (Ewald sphere, k, Q) in scene units
    N_HIGHLIGHTS = 40

    def __init__(
        self,
        crystal: "Crystal",
        energy: float,
        sphere_radius: float = 0.5,
        q_max: float | None = None,
        min_intensity: float = 1e-3,
        max_points: int | None = 3000,
        tolerance: float | None = None,
        point_radius: tuple[float, float] = (0.004, 0.02),
        limiting_sphere: bool = False,
    ) -> None:
        """Set up the scene (call :meth:`show` to render it).

        Args:
            crystal: Crystal with a UB matrix (``calc_ub``); intensities need a CIF
            energy: X-ray energy in eV
            sphere_radius: Drawn radius of the Ewald sphere at ``energy`` (sets ``scale``)
            q_max: Largest |Q| of drawn reflections in 1/Angstrom (default 2k: all reachable)
            min_intensity: Hide reflections weaker than this fraction of the strongest
            max_points: Draw at most this many (strongest) reflections
            tolerance: Excitation error for "in diffraction condition" in 1/Angstrom
                (default 1% of k)
            point_radius: (min, max) drawn point radius; radius ~ (|F|^2 / max)^(1/3)
            limiting_sphere: Also draw the limiting sphere |Q| = 2k around the origin
        """
        k = 2 * np.pi / crystal.en2lam(energy)
        self.scale = sphere_radius / k
        points = reciprocal_lattice_points(crystal, energy, q_max=q_max, min_intensity=min_intensity,
                                           max_points=max_points)
        reach = max(2 * k, points["q"].max() if len(points) else 0.0) * self.scale
        # Origin of reciprocal space high enough that everything stays above the SWIFT floor
        super().__init__(crystal, energy, origin=np.array([0.0, 0.0, reach + 0.05]))
        self._init_reflections(q_max=q_max, min_intensity=min_intensity, max_points=max_points, tolerance=tolerance)
        self._points = points
        self.sphere_radius = sphere_radius
        self.point_radius = point_radius
        self.limiting_sphere = limiting_sphere
        # Side view from the left (SwissFEL +x), centred between sphere centre and origin
        r = sphere_radius
        self.CAMERA_POSITION = np.array([-0.5 * r, 4.0 * r, 0.8 * r])
        self.CAMERA_LOOK_AT = np.array([-0.5 * r, 0.0, 0.0])
        self._create_shapes()

    def _create_shapes(self) -> None:
        import roboticstoolbox as rtb
        import spatialgeometry as sg
        import spatialmath as sm

        UB = np.asarray(self.crystal.ubcalc.UB, dtype=float)
        hkls = self.points[["h", "k", "l"]].to_numpy(dtype=float)
        # Points in the phi frame (SWIFT axes), rotated as a whole via the robot base
        local = convert(hkls @ UB.T, "diffcalc", "swift") * self.scale
        r_min, r_max = self.point_radius
        radii = np.clip(r_max * self.points["rel_intensity"].to_numpy() ** (1 / 3), r_min, r_max)
        geoms = [sg.Sphere(radius=float(r), pose=sm.SE3(*p), color=COLORS["point"]) for p, r in zip(local, radii)]
        geoms.append(sg.Sphere(radius=r_min * 1.5, color=COLORS["origin"]))
        base = rtb.Link(name="base")
        cloud = rtb.Link(rtb.ET.tz(), name="points", parent=base, geometry=geoms, qlim=(-1, 1))
        self.lattice = rtb.Robot([base, cloud], name="reciprocal_lattice")
        self.lattice.q = [0.0]

        self.highlights = [sg.Sphere(radius=r_max * 1.4, color=COLORS["highlight"]) for _ in range(self.N_HIGHLIGHTS)]
        for h in self.highlights:
            h.T = pose(np.eye(3), HIDDEN)

        step = self.SIZE_STEP
        # Spheres as opaque wireframes: translucent shapes hide what is drawn after them in SWIFT
        mesh = str(MODELS_DIR / SPHERE_WIREFRAME_FILE)
        self.sphere_pool = ShapePool(lambda r: sg.Mesh(mesh, scale=[r, r, r], color=COLORS["sphere"]), step)
        self.limiting_pool = ShapePool(lambda r: sg.Mesh(mesh, scale=[r, r, r], color=COLORS["limiting_sphere"]), step)
        self.k_i_pool = ShapePool(lambda length: arrow(COLORS["k_i"], 0.004, length), step)
        self.k_f_pool = ShapePool(lambda length: arrow(COLORS["k_f"], 0.004, length), step)
        self.q_pool = ShapePool(lambda length: arrow(COLORS["Q"], 0.005, length), step)
        # Diffracted beams per intensity class (strong thick ... weak thin), length k
        self.beam_sets = [
            ShapeSet(lambda length, r=r: arrow(DIFFRACTED_COLOR, r, length, head=0.015), self.N_HIGHLIGHTS, step)
            for r in (0.004, 0.0025, 0.0015)
        ]

        # a*, b*, c* to scale (the lattice scale is fixed)
        for i, name in enumerate(("a*", "b*", "c*")):
            self.shapes[name] = arrow(COLORS[name], 0.003, float(np.linalg.norm(UB[:, i]) * self.scale), head=0.015)

        r = self.sphere_radius
        # SwissFEL axes triad below the sphere, upstream
        self._add_frame_triad(self.origin + np.array([-2.0 * r, 0.0, -1.1 * r]), length=0.4 * r, radius=0.005)

    def _add_to_env(self) -> None:
        self.env.add(self.lattice)
        super()._add_to_env()
        for h in self.highlights:
            self.env.add(h)
        for p in (self.sphere_pool, self.limiting_pool, self.k_i_pool, self.k_f_pool, self.q_pool, *self.beam_sets):
            p.attach(self.env)

    def _update_shapes(self) -> None:
        geo = self.geometry
        sw = geo.in_frame("swift")
        o = self.origin
        s = self.scale
        k_i, k_f, Q = sw["k_i"] * s, sw["k_f"] * s, sw["Q"] * s
        k = np.linalg.norm(k_i)
        centre = o - k_i

        self.lattice.base = pose(sw["R_sample"], o)
        self.sphere_pool.show(k, pose(np.eye(3), centre))
        if self.limiting_sphere:
            self.limiting_pool.show(2 * k, pose(np.eye(3), o))
        self.k_i_pool.show(k, arrow_pose(centre, k_i))
        self.k_f_pool.show(k, arrow_pose(centre, k_f))
        q_norm = np.linalg.norm(Q)
        if q_norm > 1e-6:
            self.q_pool.show(q_norm, arrow_pose(o, Q))
        else:
            self.q_pool.show(1e-3, pose(np.eye(3), o))
        for i, name in enumerate(("a*", "b*", "c*")):
            self.shapes[name].T = arrow_pose(o, sw["reciprocal_lattice"][:, i])

        # Halos on the reflections in diffraction condition (closest first) and their
        # diffracted beams from the sphere centre
        hits = self.in_condition()
        positions = hits[["h", "k", "l"]].to_numpy(dtype=float) @ sw["reciprocal_lattice"].T * s + o
        for i, h in enumerate(self.highlights):
            h.T = pose(np.eye(3), positions[i] if i < len(positions) else HIDDEN)
        beams = hits.nlargest(self.N_HIGHLIGHTS, "intensity")
        directions = convert(beams[["kx", "ky", "kz"]].to_numpy(dtype=float), "diffcalc", "swift")
        classes = [intensity_class(r) for r in beams["rel_intensity"]]
        for c, beam_set in enumerate(self.beam_sets):
            beam_set.show(k, [arrow_pose(centre, d) for d, cls in zip(directions, classes) if cls == c])
