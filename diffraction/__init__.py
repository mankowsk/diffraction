"""Diffraction - A Python library for X-ray diffraction calculations."""

from .core.crystal import Crystal
from .phonons import PhononMode, PhononsManager

__all__ = ["Crystal", "PhononMode", "PhononsManager"]
__version__ = "0.1.0"
