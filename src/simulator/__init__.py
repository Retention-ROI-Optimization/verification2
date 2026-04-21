"""Customer behavior simulator package for Retention ROI project."""

from .config import DEFAULT_CONFIG, SimulationConfig
from .pipeline import run_simulation, run_simulation_for_dashboard

__all__ = [
    "DEFAULT_CONFIG",
    "SimulationConfig",
    "run_simulation",
    "run_simulation_for_dashboard",
]
