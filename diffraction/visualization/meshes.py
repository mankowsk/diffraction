"""Small mesh generators (binary STL) for SWIFT models.

Regenerate the committed meshes with ``python -m diffraction.visualization.meshes``.
"""

import struct
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
SPHERE_WIREFRAME_FILE = "sphere_wireframe.stl"


def write_stl(path: str | Path, triangles: np.ndarray, header: bytes = b"diffraction mesh") -> None:
    """Write triangles (N, 3, 3) as binary STL."""
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else normal
            f.write(struct.pack("<12fH", *normal, *tri.ravel(), 0))


def tube_triangles(path: np.ndarray, radius: float, sides: int = 4) -> np.ndarray:
    """Triangles of a closed tube of ``radius`` along the closed polyline ``path`` (N, 3)."""
    path = np.asarray(path, dtype=float)
    n = len(path)
    tangents = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    rings = []
    for p, t in zip(path, tangents):
        helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, helper)
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        angles = 2 * np.pi * np.arange(sides) / sides
        rings.append(p + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v))
    tris = []
    for i in range(n):
        a, b = rings[i], rings[(i + 1) % n]
        for j in range(sides):
            jn = (j + 1) % sides
            tris += [(a[j], b[j], b[jn]), (a[j], b[jn], a[jn])]
    return np.array(tris)


def sphere_wireframe_triangles(n_great_circles: int = 6, n_parallels: int = 5, tube: float = 0.006,
                               segments: int = 64) -> np.ndarray:
    """Wireframe of a unit sphere: great circles through the z poles plus parallels."""
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    tris = []
    for phi in np.pi * np.arange(n_great_circles) / n_great_circles:
        circle = np.stack([np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t)], axis=1)
        tris.append(tube_triangles(circle, tube))
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n_parallels + 2)[1:-1]:
        circle = np.stack([np.cos(lat) * np.cos(t), np.cos(lat) * np.sin(t), np.full_like(t, np.sin(lat))], axis=1)
        tris.append(tube_triangles(circle, tube))
    return np.concatenate(tris)


def write_default_meshes() -> None:
    """Regenerate ``models/sphere_wireframe.stl``."""
    write_stl(MODELS_DIR / SPHERE_WIREFRAME_FILE, sphere_wireframe_triangles(), b"diffraction sphere wireframe")


if __name__ == "__main__":
    write_default_meshes()
    print(f"Wrote {MODELS_DIR / SPHERE_WIREFRAME_FILE}")
