"""Core package for the Industrial PPE Safety Detection System."""

from .database import ViolationDatabase
from .detector import PPEDetector

__all__ = ["PPEDetector", "ViolationDatabase"]
