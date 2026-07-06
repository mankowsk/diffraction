"""Interactive 3D visualization for crystal structures and phonon modes.

This module provides tools for visualizing crystal unit cells, atomic positions,
and phonon mode animations using SWIFT and the Robotics Toolbox.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffraction.core.crystal import Crystal


class VisualizationError(Exception):
    """Base exception for visualization-related errors."""
    pass


class CrystalVisualization:
    """Interactive 3D visualization of crystal structures.

    This class provides tools for visualizing unit cells, atomic positions,
    and animating phonon modes using SWIFT (Swift Interactive Framework for
    Visualization and Tools) and the Robotics Toolbox Python.

    Attributes:
        crystal: The Crystal object to visualize
        multiplicity: List of atom multiplicities per unique site
        atoms: Dictionary mapping atom identifiers to Robot objects
        planes: Dictionary mapping plane indices to Robot objects
        scale: Scaling factor for visualization (default 0.1)
        env: SWIFT environment instance

    Example:
        >>> from diffraction import Crystal
        >>> from diffraction.visualization import CrystalVisualization
        >>> crystal = Crystal.from_cif("SrTiO3", "data/SrTiO3.cif")
        >>> vis = CrystalVisualization(crystal)
        >>> vis.create_atoms()
        >>> vis.show()
    """

    # Color mapping for common elements (RGBA format)
    COLORS: dict[str, tuple[float, float, float, float]] = {
        "O": (1.0, 0.0, 0.0, 1.0),
        "Sr": (1.0, 0.64, 0.0, 1.0),
        "Ti": (0.2, 0.2, 1.0, 1.0),
    }

    def __init__(self, crystal: "Crystal", scale: float = 0.1) -> None:
        """Initialize the CrystalVisualization with a Crystal object.

        Args:
            crystal: The Crystal instance to visualize. Must have xucalc set.
            scale: Scaling factor for atomic radii and positions (default 0.1).

        Raises:
            VisualizationError: If the crystal's xucalc attribute is None.
        """
        self.crystal = crystal
        self.multiplicity: list[int] = []
        self.atoms: dict[str, object] = {}
        self.planes: dict[int, object] = {}
        self.scale = scale
        self.env = None

        if self.crystal.xucalc is None:
            raise VisualizationError(
                "CrystalVisualization requires a Crystal with xucalc set. "
                "Load the crystal from a CIF file using Crystal.from_cif()."
            )

    def create_atoms(self) -> None:
        """Create sphere representations for all atoms in the unit cell.

        This method iterates through the lattice sites of the crystal,
        creates sphere representations for each atom at their periodic
        positions within and around the unit cell, and stores them in
        the ``atoms`` dictionary.

        The method handles multiplicity by creating copies of atoms that
        appear multiple times due to symmetry operations. Each atom is
        assigned a unique identifier based on its element type and index.

        Raises:
            VisualizationError: If SWIFT or Robotics Toolbox cannot be imported.
            RuntimeError: If the crystal structure has no lattice sites.
        """
        try:
            import swift
            import roboticstoolbox as rtb
        except ImportError as e:
            raise VisualizationError(
                "Visualization dependencies not found. Install with:\n"
                "  pip install git+https://github.com/mankowsk/swift.git\n"
                "  pip install git+https://github.com/PhilFreeman/robotics-toolbox-python.git"
            ) from e

        multiplicity = []
        prevname = ""
        num = 1

        lattice_sites = list(self.crystal.xucalc.lattice.base())

        if not lattice_sites:
            raise RuntimeError(
                f"No lattice sites found in crystal '{self.crystal.name}'. "
                "Ensure the CIF file contains valid atomic positions."
            )

        for a, pos, occ, _ in lattice_sites:
            # Track multiplicity within each unique site
            if a.name == prevname:
                num += 1
            else:
                prevname = a.name
                num = 1

            mult = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        atpos = np.array(pos) + np.array([i, j, k])
                        if all(0 <= val <= 1 for val in np.round(atpos, 2)):
                            vecpos = (
                                atpos[0] * self.crystal.xucalc.a1
                                + atpos[1] * self.crystal.xucalc.a2
                                + atpos[2] * self.crystal.xucalc.a3
                            )

                            atom_id = f"{a.name}{num}_{mult}"
                            print(f"{atom_id} {np.round(np.array(vecpos), 2)}")
                            mult +=1
                            # Determine color for the atom
                            if a.name in self.COLORS:
                                atom_color = self.COLORS[a.name]
                            elif hasattr(a, "color"):
                                atom_color = a.color
                            else:
                                atom_color = (1.0, 1.0, 1.0, 1.0)

                            sphere = self._create_atom(
                                name=atom_id,
                                position=(np.array(vecpos) + np.array([0, 0, 1])) * self.scale,
                                color=atom_color,
                                radius=a.radius / 3 if hasattr(a, "radius") else None,
                                scale=self.scale,
                            )
                            self.atoms[atom_id] = sphere

            multiplicity.append(mult)

        self.multiplicity = multiplicity

    def _create_atom(
        self,
        name: str = "atom",
        position: np.ndarray | list[float] = [0.0, 0.0, 0.0],
        color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        radius: float | None = None,
        scale: float | None = None,
    ) -> object:
        """Create a Robot object representing an atom.

        Args:
            name: Identifier for the atom
            position: Initial position as [x, y, z] coordinates
            color: RGBA color tuple (values 0-1)
            radius: Radius of the atomic sphere
            scale: Scale factor applied to geometry

        Returns:
            Robot object configured with the specified properties

        Raises:
            VisualizationError: If URDF file cannot be loaded.
        """
        try:
            import roboticstoolbox as rtb
        except ImportError:
            raise VisualizationError(
                "Robotics Toolbox not installed. Install with:\n"
                "  pip install git+https://github.com/PhilFreeman/robotics-toolbox-python.git"
            )

        try:
            links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(
                "./atom.urdf", tld="/Users/hello/switchdrive/python/tools/atomic_motions"
            )
        except Exception as e:
            raise VisualizationError(f"Failed to load atom URDF file: {e}") from e

        atom = rtb.Robot(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Configure the link geometry
        atom.links[-1].geometry[0].color = tuple(color)
        atom.q = position
        atom.q0 = position

        if radius is not None:
            atom.links[-1].geometry[0].radius = radius
        if scale is not None:
            atom.links[-1].geometry[0].radius *= scale

        return atom

    def add_atoms(self) -> None:
        """Add all atoms to the SWIFT environment.

        This method adds all previously created atom objects from ``self.atoms``
        into the visualization scene.

        Raises:
            VisualizationError: If SWIFT cannot be imported or initialized.
            RuntimeError: If show() has not been called to initialize the environment.
        """
        if self.env is None:
            raise RuntimeError(
                "SWIFT environment not initialized. Call show() before add_atoms()."
            )

        for atom in self.atoms.values():
            self.env.add(atom)

    def show(self, browser: str | None = None) -> None:
        """Display the crystal structure in a SWIFT visualization window.

        This method initializes the SWIFT environment (if not already created),
        adds all atom objects to the scene, and launches the interactive viewer.

        The visualization window allows rotation, zooming, and panning of the
        3D crystal structure.

        Raises:
            VisualizationError: If SWIFT cannot be imported or initialized.
        """
        try:
            import swift
        except ImportError:
            raise VisualizationError(
                "SWIFT not installed. Install with:\n"
                "  pip install git+https://github.com/mankowsk/swift.git"
            )

        if self.env is not None:
            self.env.reset()
        else:
            self.env = swift.Swift()
            self.env.launch(realtime=True, browser=browser)

        self.add_atoms()

    def animate(
        self,
        vectors: list[np.ndarray],
        amplitudes: list[np.ndarray],
        dt: float = 0.05,
        scale: float = 1.0,
    ) -> None:
        """Animate atomic displacements along specified vectors.

        This method creates an animation of the crystal structure by applying
        time-varying displacements to each atom based on the provided vectors
        and amplitude profiles.

        Args:
            vectors: List of displacement vectors, one per atom (shape: n_atoms x 3)
            amplitudes: List of amplitude arrays over time (n_modes x n_timepoints)
            dt: Time step in seconds for the animation (default 0.05)
            scale: Global scaling factor for displacements

        Raises:
            VisualizationError: If SWIFT environment is not initialized.
            ValueError: If vectors and amplitudes have incompatible shapes.

        Example:
            >>> # Animate along two phonon modes with sinusoidal amplitudes
            >>> n = np.linspace(0, 20*np.pi, 500)
            >>> amplitudes = 0.5 * np.array([np.sin(n), np.cos(n)])
            >>> vis.animate(vectors=[vector_x, vector_y], amplitudes=amplitudes, dt=0.01)
        """

        # Make sure that the input has the correct dimension
        if len(np.shape(amplitudes)) == 1:
            amplitudes = [amplitudes]
        if len(np.shape(vectors)) == 2:
            vectors = [vectors]

        if self.env is None:
            raise VisualizationError(
                "SWIFT environment not initialized. Call show() before animate()."
            )

        # Transpose to iterate over time steps
        for amplitude_step in np.asarray(amplitudes).T:
            # Compute total displacement as weighted sum of all mode vectors
            dq = np.sum([vec * amp for vec, amp in zip(vectors, amplitude_step)], axis=0)

            # Apply displacements to each atom
            for atom, dqa in zip(self.atoms.values(), dq):
                atom.q = atom.q0 + dqa * scale

            self.env.step(dt=dt)

    def add_white_planes(self, distance: float = 0.0) -> None:
        """Add white reference planes to the visualization scene.

        This method adds three orthogonal planes (xy, xz, yz) positioned at
        specified distances from the origin for spatial reference.

        Args:
            distance: Distance of planes from origin in Angstroms (default 0.0).
                     Planes are placed at z=distance for xy-plane and y=distance
                     for xz-plane. The yz-plane remains at x=0.

        Raises:
            VisualizationError: If SWIFT environment or Robotics Toolbox not available.
            RuntimeError: If show() has not been called to initialize the environment.
        """
        if self.env is None:
            raise RuntimeError(
                "SWIFT environment not initialized. Call show() before add_white_planes()."
            )

        try:
            import roboticstoolbox as rtb
        except ImportError:
            raise VisualizationError(
                "Robotics Toolbox not installed."
            )

        plane_configs = [
            ("x", 0),  # yz-plane at x=distance
            ("y", 1),  # xz-plane at y=distance
            ("z", 2),  # xy-plane at z=distance
        ]

        for idx, (plane_name, axis) in enumerate(plane_configs):
            try:
                links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(
                    f"./plane_{plane_name}.urdf",
                    tld="/Users/hello/switchdrive/python/tools/atomic_motions",
                )
            except Exception as e:
                raise VisualizationError(f"Failed to load plane_{plane_name}.urdf: {e}") from e

            plane = rtb.Robot(
                links,
                name=name,
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
            )

            self.planes[idx] = plane
            self.env.add(plane)

        self.set_plane_distance(distance)

    def set_plane_distance(self, distance: float) -> None:
        """Set the position of reference planes.

        Args:
            distance: Distance from origin for x and y planes (z plane remains at 0).

        Raises:
            VisualizationError: If SWIFT environment is not initialized.
        """
        if self.env is None:
            raise VisualizationError(
                "SWIFT environment not initialized."
            )

        # Set positions for xy-plane (index 2) and xz-plane (index 1)
        for idx, plane in self.planes.items():
            if idx < 2:
                plane.q[idx] = distance

        self.env.step(0.05)

    def add_lattice_vectors(self, colors: dict[str, tuple[float, float, float, float]] | None = None) -> None:
        """Add lattice vectors to the visualization scene.

        This method adds three arrow objects representing the unit cell lattice vectors
        a1, a2, and a3 originating from (0, 0, 0) and ending at their respective vector positions.

        Args:
            colors: Optional dict mapping 'a1', 'a2', 'a3' to RGBA color tuples.
                    Defaults to red for a1, green for a2, blue for a3.

        Raises:
            VisualizationError: If SWIFT environment is not initialized.
            RuntimeError: If show() has not been called to initialize the environment.
        """
        if self.env is None:
            raise RuntimeError(
                "SWIFT environment not initialized. Call show() before add_lattice_vectors()."
            )

        try:
            import roboticstoolbox as rtb
        except ImportError as e:
            raise VisualizationError(
                "Visualization dependencies not found."
            ) from e

        # Import visual primitives for arrow creation
        from diffraction.visualization.visual_primitives import arrow as create_arrow

        # Default colors: red for a1, green for a2, blue for a3
        if colors is None:
            colors = {
                "a1": (1.0, 0.0, 0.0, 1.0),   # Red
                "a2": (0.0, 1.0, 0.0, 1.0),   # Green
                "a3": (0.0, 0.0, 1.0, 1.0),   # Blue
            }

        # Define lattice vectors with origin at (0, 0, 0) and end at a1, a2, a3
        lattice_vectors = [
            ("a1", self.crystal.xucalc.a1),
            ("a2", self.crystal.xucalc.a2),
            ("a3", self.crystal.xucalc.a3),
        ]

        for vec_name, vec in lattice_vectors:
            # Scale the vector
            vec_scaled = np.array(vec) * self.scale

            # Calculate length and direction angles (spherical coordinates)
            length = np.linalg.norm(vec_scaled)

            if length < 1e-10:
                continue  # Skip zero-length vectors

            # phi = azimuth angle about z-axis (from x-axis in xy-plane)
            phi = np.arctan2(vec_scaled[1], vec_scaled[0])

            # theta = polar angle from +z axis
            theta = np.arccos(np.clip(vec_scaled[2] / length, -1.0, 1.0))

            # Create arrow with appropriate length and orientation
            # Use scaled shaft_radius proportional to vector length for better visibility
            shaft_radius = max(0.001, length * 0.01)
            cone_scale = shaft_radius * 5.0  # Maintain ~5:1 ratio

            vec_arrow = create_arrow(
                name=f"lattice_{vec_name}",
                length=length,
                shaft_radius=shaft_radius,
                cone_scale=cone_scale,
                color=colors[vec_name],
            )

            # Set q to position arrow at origin with correct orientation:
            # [x, y, z, phi, theta] = [0, 0, 0, azimuth, polar_angle]
            vec_arrow.q = np.array([0.0, 0.0, 0.0, phi, theta])

            self.env.add(vec_arrow)
