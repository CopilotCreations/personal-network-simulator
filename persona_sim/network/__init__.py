"""Network module - Social graph and interaction dynamics."""

from .graph import SocialGraph, Connection
from .dynamics import InteractionScheduler, InteractionEvent

__all__ = [
    "SocialGraph",
    "Connection",
    "InteractionScheduler",
    "InteractionEvent",
]
