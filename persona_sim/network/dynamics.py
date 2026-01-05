"""
Interaction dynamics and scheduling for persona networks.

Controls when and how personas interact with each other,
which is critical for detecting coordination patterns.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import random
import heapq


class InteractionType(Enum):
    """Types of interactions between personas."""
    POST = "post"  # Original content
    REPLY = "reply"  # Response to another's content
    SHARE = "share"  # Amplification of content
    REACT = "react"  # Simple reaction (like, etc.)


@dataclass
class InteractionEvent:
    """
    A scheduled or completed interaction event.
    
    Tracks timing, participants, and content for coordination analysis.
    """
    timestamp: datetime
    interaction_type: InteractionType
    source_id: str
    target_id: Optional[str]  # None for posts
    topic: str
    content: str = ""
    in_response_to: Optional[str] = None  # Event ID being responded to
    event_id: str = ""
    
    def __post_init__(self):
        """Initialize the event ID if not provided."""
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())
    
    def __lt__(self, other: "InteractionEvent") -> bool:
        """Compare events by timestamp for heap ordering.

        Args:
            other: Another InteractionEvent to compare against.

        Returns:
            True if this event's timestamp is earlier than the other's.
        """
        return self.timestamp < other.timestamp


@dataclass
class SchedulingConfig:
    """Configuration for interaction scheduling."""
    base_interval_minutes: float = 30.0
    interval_variance: float = 0.5  # 0.0 = fixed, 1.0 = high variance
    burst_probability: float = 0.1  # Probability of burst activity
    burst_count_range: Tuple[int, int] = (3, 10)
    quiet_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    coordination_delay_range: Tuple[float, float] = (1.0, 5.0)  # Minutes


class InteractionScheduler:
    """
    Schedules interactions between personas over simulated time.
    
    Supports:
    - Natural timing with variance
    - Burst activity patterns
    - Coordinated timing (for testing detection)
    - Quiet hours simulation
    """
    
    def __init__(
        self,
        config: Optional[SchedulingConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the interaction scheduler.

        Args:
            config: Scheduling configuration. Uses defaults if not provided.
            seed: Random seed for reproducible scheduling.
        """
        self.config = config or SchedulingConfig()
        self._rng = random.Random(seed)
        self._event_queue: List[InteractionEvent] = []
        self._completed_events: List[InteractionEvent] = []
        self._current_time: datetime = datetime.now()
    
    def set_start_time(self, start_time: datetime) -> None:
        """Set the simulation start time.

        Args:
            start_time: The datetime to set as the current simulation time.
        """
        self._current_time = start_time
    
    @property
    def current_time(self) -> datetime:
        """Get the current simulation time.

        Returns:
            The current datetime in the simulation.
        """
        return self._current_time
    
    def advance_time(self, minutes: float) -> None:
        """Advance simulation time by given minutes.

        Args:
            minutes: Number of minutes to advance the simulation clock.
        """
        self._current_time += timedelta(minutes=minutes)
    
    def schedule_event(
        self,
        interaction_type: InteractionType,
        source_id: str,
        topic: str,
        target_id: Optional[str] = None,
        content: str = "",
        delay_minutes: Optional[float] = None,
        absolute_time: Optional[datetime] = None,
    ) -> InteractionEvent:
        """Schedule a new interaction event.

        Args:
            interaction_type: Type of interaction (POST, REPLY, SHARE, REACT).
            source_id: Persona initiating the interaction.
            topic: Topic of the interaction.
            target_id: Target persona (for replies/shares). Defaults to None.
            content: Content of the interaction. Defaults to empty string.
            delay_minutes: Delay from current time. Uses natural timing if None.
            absolute_time: Specific time to schedule at. Takes precedence over delay.

        Returns:
            The scheduled InteractionEvent.
        """
        if absolute_time:
            event_time = absolute_time
        elif delay_minutes is not None:
            event_time = self._current_time + timedelta(minutes=delay_minutes)
        else:
            event_time = self._calculate_natural_time()
        
        event = InteractionEvent(
            timestamp=event_time,
            interaction_type=interaction_type,
            source_id=source_id,
            target_id=target_id,
            topic=topic,
            content=content,
        )
        
        heapq.heappush(self._event_queue, event)
        return event
    
    def _calculate_natural_time(self) -> datetime:
        """Calculate a natural-feeling next event time.

        Applies random variance to the base interval and avoids quiet hours
        to simulate realistic human posting patterns.

        Returns:
            A datetime for the next event with natural timing variance.
        """
        base = self.config.base_interval_minutes
        variance = self.config.interval_variance
        
        # Add random variance
        interval = base * (1 + (self._rng.random() - 0.5) * 2 * variance)
        interval = max(1.0, interval)  # Minimum 1 minute
        
        proposed_time = self._current_time + timedelta(minutes=interval)
        
        # Avoid quiet hours
        while proposed_time.hour in self.config.quiet_hours:
            proposed_time += timedelta(hours=1)
        
        return proposed_time
    
    def schedule_burst(
        self,
        source_id: str,
        topic: str,
        interaction_type: InteractionType = InteractionType.POST,
    ) -> List[InteractionEvent]:
        """Schedule a burst of activity from a persona.

        Bursts are characterized by rapid successive interactions,
        typically 30 seconds to 3 minutes apart.

        Args:
            source_id: The persona initiating the burst.
            topic: Topic for all burst interactions.
            interaction_type: Type of interaction. Defaults to POST.

        Returns:
            List of scheduled InteractionEvents in the burst.
        """
        count = self._rng.randint(*self.config.burst_count_range)
        events = []
        
        burst_start = self._current_time
        for i in range(count):
            delay = i * self._rng.uniform(0.5, 3.0)  # 30 seconds to 3 minutes apart
            
            event = self.schedule_event(
                interaction_type=interaction_type,
                source_id=source_id,
                topic=topic,
                absolute_time=burst_start + timedelta(minutes=delay),
            )
            events.append(event)
        
        return events
    
    def schedule_coordinated_response(
        self,
        responder_ids: List[str],
        original_event: InteractionEvent,
        topic: str,
    ) -> List[InteractionEvent]:
        """Schedule coordinated responses from multiple personas.

        This simulates coordinated behavior for detection testing.
        The timing is artificially close to detect coordination patterns.

        Args:
            responder_ids: List of persona IDs that will respond.
            original_event: The event being responded to.
            topic: Topic for the responses.

        Returns:
            List of scheduled response InteractionEvents.
        """
        events = []
        min_delay, max_delay = self.config.coordination_delay_range
        
        for responder_id in responder_ids:
            if responder_id == original_event.source_id:
                continue
            
            # Small delay range indicates coordination
            delay = self._rng.uniform(min_delay, max_delay)
            
            event = self.schedule_event(
                interaction_type=InteractionType.REPLY,
                source_id=responder_id,
                target_id=original_event.source_id,
                topic=topic,
                delay_minutes=delay,
            )
            event.in_response_to = original_event.event_id
            events.append(event)
        
        return events
    
    def schedule_organic_responses(
        self,
        responder_ids: List[str],
        original_event: InteractionEvent,
        topic: str,
        response_probability: float = 0.3,
    ) -> List[InteractionEvent]:
        """Schedule organic (non-coordinated) responses.

        Timing is more spread out with exponential distribution,
        and not all responders will respond based on probability.

        Args:
            responder_ids: List of persona IDs that may respond.
            original_event: The event being responded to.
            topic: Topic for the responses.
            response_probability: Probability each persona responds. Defaults to 0.3.

        Returns:
            List of scheduled response InteractionEvents.
        """
        events = []
        
        for responder_id in responder_ids:
            if responder_id == original_event.source_id:
                continue
            
            if self._rng.random() > response_probability:
                continue
            
            # Wide delay range for organic responses
            delay = self._rng.expovariate(1 / 60.0)  # Exponential with mean of 60 minutes
            delay = max(5.0, delay)  # At least 5 minutes
            
            event = self.schedule_event(
                interaction_type=InteractionType.REPLY,
                source_id=responder_id,
                target_id=original_event.source_id,
                topic=topic,
                delay_minutes=delay,
            )
            event.in_response_to = original_event.event_id
            events.append(event)
        
        return events
    
    def get_next_event(self) -> Optional[InteractionEvent]:
        """Get the next scheduled event and remove it from the queue.

        Returns:
            The next InteractionEvent, or None if no events are scheduled.
        """
        if not self._event_queue:
            return None
        
        return heapq.heappop(self._event_queue)
    
    def peek_next_event(self) -> Optional[InteractionEvent]:
        """Peek at the next event without removing it from the queue.

        Returns:
            The next InteractionEvent, or None if the queue is empty.
        """
        if not self._event_queue:
            return None
        return self._event_queue[0]
    
    def process_next_event(self) -> Optional[InteractionEvent]:
        """Process the next event and advance simulation time.

        Removes the event from the queue, advances the clock to its
        timestamp, and adds it to the completed events list.

        Returns:
            The processed InteractionEvent, or None if the queue is empty.
        """
        event = self.get_next_event()
        if event:
            self._current_time = event.timestamp
            self._completed_events.append(event)
        return event
    
    def process_events_until(self, end_time: datetime) -> List[InteractionEvent]:
        """Process all events until the given time.

        Args:
            end_time: The datetime to process events up to (inclusive).

        Returns:
            List of all processed InteractionEvents.
        """
        processed = []
        
        while self._event_queue and self._event_queue[0].timestamp <= end_time:
            event = self.process_next_event()
            if event:
                processed.append(event)
        
        self._current_time = end_time
        return processed
    
    def get_completed_events(self) -> List[InteractionEvent]:
        """Get all completed events.

        Returns:
            A copy of the list of all completed InteractionEvents.
        """
        return self._completed_events.copy()
    
    def get_events_by_persona(self, persona_id: str) -> List[InteractionEvent]:
        """Get all completed events by a specific persona.

        Args:
            persona_id: The ID of the persona to filter by.

        Returns:
            List of InteractionEvents where the persona is the source.
        """
        return [e for e in self._completed_events if e.source_id == persona_id]
    
    def get_events_by_topic(self, topic: str) -> List[InteractionEvent]:
        """Get all completed events on a specific topic.

        Args:
            topic: The topic string to filter by.

        Returns:
            List of InteractionEvents matching the specified topic.
        """
        return [e for e in self._completed_events if e.topic == topic]
    
    def get_timing_statistics(self, persona_id: str) -> Dict[str, float]:
        """Get timing statistics for a persona.

        Calculates average and standard deviation of intervals between
        consecutive events. Useful for detecting coordination patterns.

        Args:
            persona_id: The ID of the persona to analyze.

        Returns:
            Dictionary containing:
                - avg_interval_minutes: Average time between events.
                - std_interval_minutes: Standard deviation of intervals.
                - total_events: Total number of events by this persona.
        """
        events = self.get_events_by_persona(persona_id)
        
        if len(events) < 2:
            return {
                "avg_interval_minutes": 0.0,
                "std_interval_minutes": 0.0,
                "total_events": len(events),
            }
        
        # Calculate intervals between consecutive events
        intervals = []
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i in range(1, len(sorted_events)):
            delta = sorted_events[i].timestamp - sorted_events[i - 1].timestamp
            intervals.append(delta.total_seconds() / 60.0)
        
        avg = sum(intervals) / len(intervals)
        variance = sum((x - avg) ** 2 for x in intervals) / len(intervals)
        std = variance ** 0.5
        
        return {
            "avg_interval_minutes": avg,
            "std_interval_minutes": std,
            "total_events": len(events),
        }
    
    @property
    def pending_count(self) -> int:
        """Get the number of pending events in the queue.

        Returns:
            Count of events waiting to be processed.
        """
        return len(self._event_queue)
    
    @property
    def completed_count(self) -> int:
        """Get the number of completed events.

        Returns:
            Count of events that have been processed.
        """
        return len(self._completed_events)
    
    def clear(self) -> None:
        """Clear all events from both the queue and completed list."""
        self._event_queue.clear()
        self._completed_events.clear()
    
    def __repr__(self) -> str:
        """Return a string representation of the scheduler.

        Returns:
            String showing pending and completed event counts.
        """
        return f"InteractionScheduler(pending={self.pending_count}, completed={self.completed_count})"
