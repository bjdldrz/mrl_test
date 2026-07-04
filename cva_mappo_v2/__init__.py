"""
CVA-MAPPO v2.

This package is intentionally separate from the legacy environment code.  It
implements task-centered candidate assignment, typed fixed-size action slots,
and event-triggered candidate repair while reusing the existing single-satellite
physics and MAPPO trainer.
"""

from .config import CVAMAPPOV2Config, CandidateSlotConfig
from .env import CVAMAPPOV2Env

__all__ = [
    "CVAMAPPOV2Config",
    "CandidateSlotConfig",
    "CVAMAPPOV2Env",
]
