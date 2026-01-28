"""pose_extract.track_solution package.

Expose core classes for convenient imports:
- TrackManager (manager_refactored)
- TrackRepository (repository)
- TrackRecord, TrackState
- TrackVisualizer (visualization)
"""

from .manager import TrackManager
from .repository import TrackRepository
from .record import TrackRecord, TrackState
from .visualization import TrackVisualizer

__all__ = [
    "TrackManager",
    "TrackRepository",
    "TrackRecord",
    "TrackState",
    "TrackVisualizer",
]