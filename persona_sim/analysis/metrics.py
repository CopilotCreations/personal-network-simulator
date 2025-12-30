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
        """Record metrics for a simulation step."""
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
        """Record a single interaction event."""
        self._persona_interactions[event.source_id] += 1
        self._topic_interaction_counts[event.topic] += 1
        
        if event.target_id:
            self._persona_interactions[event.target_id] += 1
    
    def record_belief_change(self, persona_id: str, topic: str) -> None:
        """Record a belief change."""
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
        """Finalize metrics collection and return aggregated metrics."""
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
        """Calculate average of a list."""
        return sum(values) / len(values) if values else 0.0
    
    def get_persona_activity(self, persona_id: str) -> Dict[str, int]:
        """Get activity metrics for a specific persona."""
        return {
            "interactions": self._persona_interactions.get(persona_id, 0),
            "belief_changes": self._persona_belief_changes.get(persona_id, 0),
        }
    
    def get_top_active_personas(self, n: int = 10) -> List[tuple]:
        """Get the most active personas."""
        sorted_personas = sorted(
            self._persona_interactions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_personas[:n]
    
    def get_top_topics(self, n: int = 10) -> List[tuple]:
        """Get the most discussed topics."""
        sorted_topics = sorted(
            self._topic_interaction_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_topics[:n]
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get the timeline of step metrics."""
        return self._timeline.copy()
    
    def get_activity_distribution(self) -> Dict[str, Any]:
        """Get distribution statistics for persona activity."""
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
        """Compare this collector's metrics with another."""
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
        """Export timeline data to CSV."""
        import csv
        
        if not self._timeline:
            return
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._timeline[0].keys())
            writer.writeheader()
            writer.writerows(self._timeline)
    
    def __repr__(self) -> str:
        return f"MetricsCollector(id={self.simulation_id}, steps={self._step_count}, interactions={self._interaction_count})"
