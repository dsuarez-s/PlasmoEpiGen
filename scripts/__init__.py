# scripts/__init__.py

__version__ = "0.1.0"

# Re‑export the core model class
from .model.malaria_eg_model import MalariaEGModel

__all__ = ["MalariaEGModel"]