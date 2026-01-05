"""
Metrics collection for simulation analysis.

Collects and aggregates various metrics about simulation runs
for comparison and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json

from ..simulation.engine import SimulationState
from ..network.dynamics import InteractionEvent


@dataclass
class SimulationMetrics:
    """Aggregated metrics for a simulation run."""
    # Basic stats
    simulation_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    total_interactions: int
    total_belief_changes: int
    
    # Persona stats
    persona_count: int
    coordinated_persona_count: int
    
    # Network stats
    connection_count: int
    graph_density: float
    avg_clustering: float
    
    # Activity stats
    interactions_per_step: float
    interactions_per_persona: float
    
    # Convergence stats
    topics_tracked: int
    topics_with_consensus: int
    avg_convergence_rate: float
    
    # Coordination detection
    coordination_signals_detected: int
    strong_signals: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all metric fields with
                JSON-serializable values.
        """
        return {
            "simulation_id": self.simulation_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_steps": self.total_steps,
            "total_interactions": self.total_interactions,
            "total_belief_changes": self.total_belief_changes,
            "persona_count": self.persona_count,
            "coordinated_persona_count": self.coordinated_persona_count,
            "connection_count": self.connection_count,
            "graph_density": self.graph_density,
            "avg_clustering": self.avg_clustering,
            "interactions_per_step": self.interactions_per_step,
            "interactions_per_persona": self.interactions_per_persona,
            "topics_tracked": self.topics_tracked,
            "topics_with_consensus": self.topics_with_consensus,
            "avg_convergence_rate": self.avg_convergence_rate,
            "coordination_signals_detected": self.coordination_signals_detected,
            "strong_signals": self.strong_signals,
        }
    
    def to_json(self) -> str:
        """Convert metrics to a JSON string.

        Returns:
            str: JSON-formatted string representation of the metrics.
        """
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """
    Collects metrics during simulation runs.
    
    Tracks:
    - Per-step metrics
    - Per-persona metrics
    - Aggregate statistics
    """
    
    def __init__(self, simulation_id: Optional[str] = None):
        """Initialize a new metrics collector.

        Args:
            simulation_id: Optional unique identifier for the simulation.
                If not provided, a random 8-character UUID will be generated.
        """
        import uuid
        self.simulation_id = simulation_id or str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Counters
        self._step_count = 0
        self._interaction_count = 0
        self._belief_change_count = 0
        
        # Per-step tracking
        self._interactions_per_step: List[int] = []
        self._belief_changes_per_step: List[int] = []
        
        # Per-persona tracking
        self._persona_interactions: Dict[str, int] = defaultdict(int)
        self._persona_belief_changes: Dict[str, int] = defaultdict(int)
        
        # Topic tracking
        self._topic_interaction_counts: Dict[str, int] = defaultdict(int)
        
        # Time series data
        self._timeline: List[Dict[str, Any]] = []
    
    def record_step(
        self,
        state: SimulationState,
        interactions_this_step: int = 0,
        belief_changes_this_step: int = 0,
    ) -> None:
        """Record metrics for a simulation step.

        Args:
            state: Current simulation state containing step count and totals.
            interactions_this_step: Number of interactions that occurred
                during this step.
            belief_changes_this_step: Number of belief changes that occurred
                during this step.
        """
        self._step_count = state.step_count
        self._interaction_count = state.total_interactions
        self._belief_change_count = state.belief_changes
        
        self._interactions_per_step.append(interactions_this_step)
        self._belief_changes_per_step.append(belief_changes_this_step)
        
        # Timeline entry
        self._timeline.append({
            "step": state.step_count,
            "time": state.current_time.isoformat(),
            "interactions": interactions_this_step,
            "belief_changes": belief_changes_this_step,
            "total_interactions": state.total_interactions,
        })
    
    def record_interaction(self, event: InteractionEvent) -> None:
        """Record a single interaction event.

        Args:
            event: The interaction event to record. Updates interaction
                counts for the source persona, target persona (if any),
                and the topic involved.
        """
        self._persona_interactions[event.source_id] += 1
        self._topic_interaction_counts[event.topic] += 1
        
        if event.target_id:
            self._persona_interactions[event.target_id] += 1
    
    def record_belief_change(self, persona_id: str, topic: str) -> None:
        """Record a belief change.

        Args:
            persona_id: Identifier of the persona whose belief changed.
            topic: The topic for which the belief changed.
        """
        self._persona_belief_changes[persona_id] += 1
    
    def finalize(
        self,
        persona_count: int,
        coordinated_count: int,
        connection_count: int,
        graph_density: float,
        avg_clustering: float,
        topics_tracked: int,
        topics_with_consensus: int,
        avg_convergence_rate: float,
        coordination_signals: int,
        strong_signals: int,
    ) -> SimulationMetrics:
        """Finalize metrics collection and return aggregated metrics.

        Args:
            persona_count: Total number of personas in the simulation.
            coordinated_count: Number of personas exhibiting coordinated behavior.
            connection_count: Total number of connections in the network.
            graph_density: Density of the network graph (0.0 to 1.0).
            avg_clustering: Average clustering coefficient of the network.
            topics_tracked: Number of topics being tracked for convergence.
            topics_with_consensus: Number of topics that reached consensus.
            avg_convergence_rate: Average rate of belief convergence across topics.
            coordination_signals: Total coordination signals detected.
            strong_signals: Number of strong coordination signals detected.

        Returns:
            SimulationMetrics: Aggregated metrics for the completed simulation run.
        """
        self.end_time = datetime.now()
        
        return SimulationMetrics(
            simulation_id=self.simulation_id,
            start_time=self.start_time,
            end_time=self.end_time,
            total_steps=self._step_count,
            total_interactions=self._interaction_count,
            total_belief_changes=self._belief_change_count,
            persona_count=persona_count,
            coordinated_persona_count=coordinated_count,
            connection_count=connection_count,
            graph_density=graph_density,
            avg_clustering=avg_clustering,
            interactions_per_step=self._avg(self._interactions_per_step),
            interactions_per_persona=self._interaction_count / max(1, persona_count),
            topics_tracked=topics_tracked,
            topics_with_consensus=topics_with_consensus,
            avg_convergence_rate=avg_convergence_rate,
            coordination_signals_detected=coordination_signals,
            strong_signals=strong_signals,
        )
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average of a list.

        Args:
            values: List of numeric values to average.

        Returns:
            float: The arithmetic mean of the values, or 0.0 if the list is empty.
        """
        return sum(values) / len(values) if values else 0.0
    
    def get_persona_activity(self, persona_id: str) -> Dict[str, int]:
        """Get activity metrics for a specific persona.

        Args:
            persona_id: Identifier of the persona to query.

        Returns:
            Dict[str, int]: Dictionary containing 'interactions' and
                'belief_changes' counts for the persona.
        """
        return {
            "interactions": self._persona_interactions.get(persona_id, 0),
            "belief_changes": self._persona_belief_changes.get(persona_id, 0),
        }
    
    def get_top_active_personas(self, n: int = 10) -> List[tuple]:
        """Get the most active personas.

        Args:
            n: Maximum number of personas to return.

        Returns:
            List[tuple]: List of (persona_id, interaction_count) tuples,
                sorted by interaction count in descending order.
        """
        sorted_personas = sorted(
            self._persona_interactions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_personas[:n]
    
    def get_top_topics(self, n: int = 10) -> List[tuple]:
        """Get the most discussed topics.

        Args:
            n: Maximum number of topics to return.

        Returns:
            List[tuple]: List of (topic, interaction_count) tuples,
                sorted by interaction count in descending order.
        """
        sorted_topics = sorted(
            self._topic_interaction_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_topics[:n]
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get the timeline of step metrics.

        Returns:
            List[Dict[str, Any]]: Copy of the timeline data, where each entry
                contains step number, timestamp, and interaction/belief change
                counts for that step.
        """
        return self._timeline.copy()
    
    def get_activity_distribution(self) -> Dict[str, Any]:
        """Get distribution statistics for persona activity.

        Returns:
            Dict[str, Any]: Dictionary containing 'min', 'max', 'mean', and
                'std' (standard deviation) of interaction counts across
                all personas.
        """
        interactions = list(self._persona_interactions.values())
        
        if not interactions:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        
        mean = sum(interactions) / len(interactions)
        variance = sum((x - mean) ** 2 for x in interactions) / len(interactions)
        
        return {
            "min": min(interactions),
            "max": max(interactions),
            "mean": mean,
            "std": variance ** 0.5,
        }
    
    def compare_with(self, other: "MetricsCollector") -> Dict[str, Any]:
        """Compare this collector's metrics with another.

        Args:
            other: Another MetricsCollector instance to compare against.

        Returns:
            Dict[str, Any]: Dictionary containing comparison ratios and
                differences including 'interaction_ratio', 'belief_change_ratio',
                'step_ratio', and 'avg_interactions_diff'.
        """
        return {
            "interaction_ratio": self._interaction_count / max(1, other._interaction_count),
            "belief_change_ratio": self._belief_change_count / max(1, other._belief_change_count),
            "step_ratio": self._step_count / max(1, other._step_count),
            "avg_interactions_diff": (
                self._avg(self._interactions_per_step) -
                other._avg(other._interactions_per_step)
            ),
        }
    
    def export_to_csv(self, filepath: str) -> None:
        """Export timeline data to CSV.

        Args:
            filepath: Path to the output CSV file. If the timeline is empty,
                no file will be created.
        """
        import csv
        
        if not self._timeline:
            return
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._timeline[0].keys())
            writer.writeheader()
            writer.writerows(self._timeline)
    
    def __repr__(self) -> str:
        """Return a string representation of the collector.

        Returns:
            str: String containing the simulation ID, step count, and
                interaction count.
        """
        return f"MetricsCollector(id={self.simulation_id}, steps={self._step_count}, interactions={self._interaction_count})"
