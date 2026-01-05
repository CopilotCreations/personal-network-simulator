"""
Time-step simulation loop for persona networks.

Orchestrates interactions between personas over simulated time,
tracking state changes for analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import random

from ..agents.persona import Persona
from ..agents.memory import Memory
from ..agents.style import LinguisticStyle, StyleConstraints
from ..network.graph import SocialGraph, Connection
from ..network.dynamics import (
    InteractionScheduler,
    InteractionEvent,
    InteractionType,
    SchedulingConfig,
)


class SimulationPhase(Enum):
    """Phases of a simulation run."""
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    # Time settings
    start_time: datetime = field(default_factory=datetime.now)
    duration_hours: float = 24.0
    time_step_minutes: float = 15.0
    
    # Interaction settings
    interactions_per_step: int = 10
    response_probability: float = 0.3
    
    # Coordination injection (for testing detection)
    inject_coordination: bool = False
    coordinated_persona_fraction: float = 0.2
    coordination_topics: List[str] = field(default_factory=list)
    
    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class SimulationState:
    """Current state of a simulation."""
    phase: SimulationPhase = SimulationPhase.SETUP
    current_time: datetime = field(default_factory=datetime.now)
    step_count: int = 0
    total_interactions: int = 0
    belief_changes: int = 0


class SimulationEngine:
    """
    Main simulation engine for persona network interactions.
    
    Runs a time-step simulation where:
    1. Personas post content based on their beliefs
    2. Other personas respond based on connections and beliefs
    3. Beliefs update based on interactions
    4. All events are logged for analysis
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
    ):
        """Initialize the simulation engine.

        Args:
            config: Configuration for the simulation. If None, uses defaults.
        """
        self.config = config or SimulationConfig()
        self._rng = random.Random(self.config.seed)
        
        # Core components
        self._personas: Dict[str, Persona] = {}
        self._memories: Dict[str, Memory] = {}
        self._styles: Dict[str, LinguisticStyle] = {}
        self._graph: SocialGraph = SocialGraph()
        self._scheduler: InteractionScheduler = InteractionScheduler(
            seed=self.config.seed
        )
        
        # State tracking
        self._state = SimulationState()
        self._state.current_time = self.config.start_time
        self._scheduler.set_start_time(self.config.start_time)
        
        # Event hooks
        self._on_interaction: List[Callable[[InteractionEvent], None]] = []
        self._on_step: List[Callable[[SimulationState], None]] = []
        
        # Coordinated personas (for testing detection)
        self._coordinated_ids: List[str] = []
    
    def add_persona(
        self,
        persona: Persona,
        style_constraints: Optional[StyleConstraints] = None,
        is_coordinated: bool = False,
    ) -> None:
        """Add a persona to the simulation.

        Args:
            persona: The persona to add.
            style_constraints: Optional linguistic style constraints for the persona.
            is_coordinated: Whether this persona is part of a coordinated group.
        """
        self._personas[persona.id] = persona
        self._memories[persona.id] = Memory()
        
        if style_constraints:
            self._styles[persona.id] = LinguisticStyle(style_constraints)
            self._styles[persona.id].set_seed(hash(persona.id))
        else:
            self._styles[persona.id] = LinguisticStyle(StyleConstraints())
        
        self._graph.add_node(persona.id)
        
        if is_coordinated:
            self._coordinated_ids.append(persona.id)
    
    def add_connection(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.5,
    ) -> None:
        """Add a connection between two personas.

        Args:
            source_id: ID of the source persona.
            target_id: ID of the target persona.
            strength: Connection strength between 0 and 1. Defaults to 0.5.
        """
        from ..network.graph import ConnectionType
        self._graph.add_connection(source_id, target_id, ConnectionType.MUTUAL, strength)
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID.

        Args:
            persona_id: The unique identifier of the persona.

        Returns:
            The persona if found, None otherwise.
        """
        return self._personas.get(persona_id)
    
    def get_memory(self, persona_id: str) -> Optional[Memory]:
        """Get a persona's memory.

        Args:
            persona_id: The unique identifier of the persona.

        Returns:
            The persona's memory if found, None otherwise.
        """
        return self._memories.get(persona_id)
    
    def get_style(self, persona_id: str) -> Optional[LinguisticStyle]:
        """Get a persona's linguistic style.

        Args:
            persona_id: The unique identifier of the persona.

        Returns:
            The persona's linguistic style if found, None otherwise.
        """
        return self._styles.get(persona_id)
    
    @property
    def graph(self) -> SocialGraph:
        """Get the social graph.

        Returns:
            The social graph representing connections between personas.
        """
        return self._graph
    
    @property
    def state(self) -> SimulationState:
        """Get current simulation state.

        Returns:
            The current state of the simulation.
        """
        return self._state
    
    @property
    def personas(self) -> List[Persona]:
        """Get all personas.

        Returns:
            List of all personas in the simulation.
        """
        return list(self._personas.values())
    
    def on_interaction(self, callback: Callable[[InteractionEvent], None]) -> None:
        """Register a callback for interaction events.

        Args:
            callback: Function to call when an interaction event occurs.
        """
        self._on_interaction.append(callback)
    
    def on_step(self, callback: Callable[[SimulationState], None]) -> None:
        """Register a callback for step completion.

        Args:
            callback: Function to call when a simulation step completes.
        """
        self._on_step.append(callback)
    
    def run(self) -> SimulationState:
        """Run the full simulation.

        Executes simulation steps until the configured duration is reached
        or the simulation is paused.

        Returns:
            The final simulation state after completion.
        """
        self._state.phase = SimulationPhase.RUNNING
        end_time = self.config.start_time + timedelta(hours=self.config.duration_hours)
        
        while self._state.current_time < end_time:
            self._run_step()
            
            if self._state.phase != SimulationPhase.RUNNING:
                break
        
        self._state.phase = SimulationPhase.COMPLETED
        return self._state
    
    def run_steps(self, n_steps: int) -> SimulationState:
        """Run a specific number of simulation steps.

        Args:
            n_steps: Number of steps to run.

        Returns:
            The simulation state after running the specified steps.
        """
        self._state.phase = SimulationPhase.RUNNING
        
        for _ in range(n_steps):
            self._run_step()
            
            if self._state.phase != SimulationPhase.RUNNING:
                break
        
        return self._state
    
    def pause(self) -> None:
        """Pause the simulation.

        Sets the simulation phase to PAUSED, stopping step execution.
        """
        self._state.phase = SimulationPhase.PAUSED
    
    def resume(self) -> None:
        """Resume a paused simulation.

        Resumes execution only if the simulation is currently paused.
        """
        if self._state.phase == SimulationPhase.PAUSED:
            self._state.phase = SimulationPhase.RUNNING
    
    def _run_step(self) -> None:
        """Execute a single simulation step.

        Generates new interactions, processes scheduled events,
        advances time, and decays connection strengths.
        """
        # Generate new interactions for this step
        self._generate_interactions()
        
        # Process scheduled events
        step_end = self._state.current_time + timedelta(
            minutes=self.config.time_step_minutes
        )
        events = self._scheduler.process_events_until(step_end)
        
        # Handle each event
        for event in events:
            self._handle_event(event)
        
        # Advance time
        self._state.current_time = step_end
        self._state.step_count += 1
        
        # Decay connection strengths
        self._graph.decay_all_connections(0.999)
        
        # Call step callbacks
        for callback in self._on_step:
            callback(self._state)
    
    def _generate_interactions(self) -> None:
        """Generate new interactions for the current step.

        Selects random personas to post content and schedules responses
        from followers. Handles both coordinated and organic responses.
        """
        # Select random personas to post
        active_personas = self._rng.sample(
            list(self._personas.values()),
            min(self.config.interactions_per_step, len(self._personas)),
        )
        
        for persona in active_personas:
            # Choose a topic from beliefs or random
            if persona.beliefs:
                topic = self._rng.choice(list(persona.beliefs.keys()))
            else:
                topic = "general"
            
            # Schedule a post
            event = self._scheduler.schedule_event(
                interaction_type=InteractionType.POST,
                source_id=persona.id,
                topic=topic,
                content=persona.generate_response(topic),
            )
            
            # Schedule responses from followers
            followers = self._graph.get_followers(persona.id)
            
            # Check if this should trigger coordinated response
            if (
                self.config.inject_coordination
                and persona.id in self._coordinated_ids
                and topic in self.config.coordination_topics
            ):
                # Coordinated response from other coordinated personas
                coordinated_responders = [
                    pid for pid in self._coordinated_ids
                    if pid != persona.id and pid in followers
                ]
                if coordinated_responders:
                    self._scheduler.schedule_coordinated_response(
                        coordinated_responders,
                        event,
                        topic,
                    )
            
            # Organic responses from non-coordinated followers
            organic_responders = [
                pid for pid in followers
                if pid not in self._coordinated_ids
            ]
            if organic_responders:
                self._scheduler.schedule_organic_responses(
                    organic_responders,
                    event,
                    topic,
                    self.config.response_probability,
                )
    
    def _handle_event(self, event: InteractionEvent) -> None:
        """Handle a single interaction event.

        Applies linguistic style, records memories, updates beliefs,
        and strengthens connections based on the interaction.

        Args:
            event: The interaction event to process.
        """
        source = self._personas.get(event.source_id)
        if not source:
            return
        
        # Apply linguistic style to content
        style = self._styles.get(event.source_id)
        if style and event.content:
            event.content = style.apply_style(event.content)
        
        # Record interaction in source's memory
        source_memory = self._memories.get(event.source_id)
        if source_memory:
            source_memory.add_memory(
                interaction_type="sent",
                source_id=event.source_id,
                topic=event.topic,
                content_summary=event.content[:100] if event.content else "",
                sentiment=self._estimate_sentiment(event.content),
            )
        
        # If this is a reply, update beliefs
        if event.target_id and event.target_id in self._personas:
            target = self._personas[event.target_id]
            target_memory = self._memories.get(event.target_id)
            
            # Target receives the message
            if target_memory:
                target_memory.add_memory(
                    interaction_type="received",
                    source_id=event.source_id,
                    topic=event.topic,
                    content_summary=event.content[:100] if event.content else "",
                    sentiment=self._estimate_sentiment(event.content),
                )
            
            # Update target's belief based on source's position
            source_belief = source.get_belief(event.topic)
            if source_belief:
                old_belief = target.get_belief(event.topic)
                target.update_belief(
                    event.topic,
                    source_belief.position,
                    source.profile.credibility_weight,
                )
                new_belief = target.get_belief(event.topic)
                
                if old_belief is None or abs(old_belief.position - new_belief.position) > 0.01:
                    self._state.belief_changes += 1
            
            # Strengthen connection
            connection = self._graph.get_connection(event.source_id, event.target_id)
            if connection:
                connection.record_interaction()
        
        source.record_interaction()
        self._state.total_interactions += 1
        
        # Call interaction callbacks
        for callback in self._on_interaction:
            callback(event)
    
    def _estimate_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment estimation.

        Args:
            text: The text to analyze for sentiment.

        Returns:
            A sentiment score between -1 (negative) and 1 (positive).
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        positive_words = ["support", "agree", "good", "great", "beneficial", "important", "embrace"]
        negative_words = ["concern", "problem", "issue", "bad", "harmful", "cautious", "drawback"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def get_all_events(self) -> List[InteractionEvent]:
        """Get all completed events.

        Returns:
            List of all interaction events that have been processed.
        """
        return self._scheduler.get_completed_events()
    
    def get_events_by_topic(self, topic: str) -> List[InteractionEvent]:
        """Get all events for a specific topic.

        Args:
            topic: The topic to filter events by.

        Returns:
            List of interaction events matching the specified topic.
        """
        return self._scheduler.get_events_by_topic(topic)
    
    def export_state(self) -> Dict[str, Any]:
        """Export current simulation state for analysis.

        Returns:
            Dictionary containing simulation metrics and state information.
        """
        return {
            "phase": self._state.phase.value,
            "current_time": self._state.current_time.isoformat(),
            "step_count": self._state.step_count,
            "total_interactions": self._state.total_interactions,
            "belief_changes": self._state.belief_changes,
            "persona_count": len(self._personas),
            "connection_count": self._graph.edge_count,
            "coordinated_count": len(self._coordinated_ids),
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the simulation engine.

        Returns:
            String showing persona count and current phase.
        """
        return f"SimulationEngine(personas={len(self._personas)}, phase={self._state.phase.value})"
