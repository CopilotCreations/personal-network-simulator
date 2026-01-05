"""
Topic and belief tracking for narrative analysis.

Tracks how narratives spread and evolve across the persona network,
measuring convergence and divergence over time.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import math


@dataclass
class BeliefSnapshot:
    """
    A snapshot of belief states across all personas at a point in time.
    
    Used to track how beliefs change over the simulation.
    """
    timestamp: datetime
    topic: str
    belief_positions: Dict[str, float]  # persona_id -> position
    belief_confidences: Dict[str, float]  # persona_id -> confidence
    
    @property
    def mean_position(self) -> float:
        """Calculate the average belief position across all personas.
        
        Returns:
            float: The mean of all belief positions, or 0.0 if no positions exist.
        """
        if not self.belief_positions:
            return 0.0
        return sum(self.belief_positions.values()) / len(self.belief_positions)
    
    @property
    def position_variance(self) -> float:
        """Calculate the variance of belief positions.
        
        Returns:
            float: The variance of belief positions, or 0.0 if fewer than 2 positions.
        """
        if len(self.belief_positions) < 2:
            return 0.0
        mean = self.mean_position
        return sum((p - mean) ** 2 for p in self.belief_positions.values()) / len(self.belief_positions)
    
    @property
    def consensus_score(self) -> float:
        """Measure the degree of consensus in beliefs.
        
        Based on inverse of variance normalized to [0, 1].
        
        Returns:
            float: A value from 0 (no consensus) to 1 (full consensus).
        """
        # Max variance is 1.0 (positions range from -1 to 1)
        return 1.0 - min(1.0, self.position_variance)
    
    def get_clusters(self, threshold: float = 0.3) -> List[Set[str]]:
        """Identify clusters of personas with similar beliefs.
        
        Uses simple clustering based on position distance.
        
        Args:
            threshold: Maximum position difference for personas to be in the same
                cluster. Defaults to 0.3.
        
        Returns:
            List[Set[str]]: A list of sets, where each set contains persona IDs
                belonging to the same cluster.
        """
        clusters: List[Set[str]] = []
        assigned = set()
        
        personas = list(self.belief_positions.keys())
        
        for persona_id in personas:
            if persona_id in assigned:
                continue
            
            position = self.belief_positions[persona_id]
            cluster = {persona_id}
            assigned.add(persona_id)
            
            for other_id in personas:
                if other_id in assigned:
                    continue
                
                other_position = self.belief_positions[other_id]
                if abs(position - other_position) <= threshold:
                    cluster.add(other_id)
                    assigned.add(other_id)
            
            clusters.append(cluster)
        
        return clusters


@dataclass
class Narrative:
    """
    A narrative is a topic with associated beliefs and their evolution.
    
    Tracks:
    - Who holds what position
    - How positions change over time
    - Sources of influence
    """
    topic: str
    created_at: datetime = field(default_factory=datetime.now)
    snapshots: List[BeliefSnapshot] = field(default_factory=list)
    interaction_count: int = 0
    propagation_paths: List[Tuple[str, str, datetime]] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: BeliefSnapshot) -> None:
        """Add a belief snapshot to the narrative history.
        
        Args:
            snapshot: The BeliefSnapshot to add to the timeline.
        """
        self.snapshots.append(snapshot)
    
    def record_propagation(self, source_id: str, target_id: str, timestamp: datetime) -> None:
        """Record a belief propagation event from source to target persona.
        
        Args:
            source_id: The persona ID who influenced the belief change.
            target_id: The persona ID whose belief was influenced.
            timestamp: When the propagation occurred.
        """
        self.propagation_paths.append((source_id, target_id, timestamp))
        self.interaction_count += 1
    
    @property
    def latest_snapshot(self) -> Optional[BeliefSnapshot]:
        """Get the most recent belief snapshot.
        
        Returns:
            Optional[BeliefSnapshot]: The last snapshot in the timeline, or None
                if no snapshots exist.
        """
        return self.snapshots[-1] if self.snapshots else None
    
    def get_convergence_rate(self) -> float:
        """Calculate how quickly beliefs are converging or diverging.
        
        Compares variance between first and last snapshots, normalized by time.
        
        Returns:
            float: Rate of convergence per hour. Positive values indicate
                converging beliefs, negative values indicate diverging beliefs,
                and 0 indicates stable beliefs.
        """
        if len(self.snapshots) < 2:
            return 0.0
        
        # Compare first and last variance
        first_var = self.snapshots[0].position_variance
        last_var = self.snapshots[-1].position_variance
        
        # Normalize by time
        time_delta = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
        if time_delta == 0:
            return 0.0
        
        # Positive when variance decreases (converging)
        return (first_var - last_var) / (time_delta / 3600)  # Per hour
    
    def get_influential_personas(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Identify the most influential personas for this narrative.
        
        Influence is measured by how often a persona appears as a source
        in belief propagation events.
        
        Args:
            top_n: Maximum number of personas to return. Defaults to 5.
        
        Returns:
            List[Tuple[str, int]]: A list of tuples containing (persona_id, count)
                sorted by influence count in descending order.
        """
        source_counts: Dict[str, int] = defaultdict(int)
        
        for source_id, _, _ in self.propagation_paths:
            source_counts[source_id] += 1
        
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_sources[:top_n]


class NarrativeTracker:
    """
    Tracks narratives across the persona network.
    
    Monitors:
    - Narrative emergence and spread
    - Belief convergence/divergence
    - Influence patterns
    """
    
    def __init__(self):
        """Initialize the NarrativeTracker with empty tracking data."""
        self._narratives: Dict[str, Narrative] = {}
        self._persona_beliefs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._snapshot_interval: int = 10  # Steps between snapshots
        self._step_count: int = 0
    
    def register_topic(self, topic: str) -> Narrative:
        """Register a new topic to track.
        
        Args:
            topic: The topic name to register.
        
        Returns:
            Narrative: The Narrative object for the topic (existing or newly created).
        """
        if topic not in self._narratives:
            self._narratives[topic] = Narrative(topic=topic)
        return self._narratives[topic]
    
    def get_narrative(self, topic: str) -> Optional[Narrative]:
        """Get a tracked narrative by topic name.
        
        Args:
            topic: The topic name to look up.
        
        Returns:
            Optional[Narrative]: The Narrative object if found, None otherwise.
        """
        return self._narratives.get(topic)
    
    @property
    def topics(self) -> List[str]:
        """Get all tracked topic names.
        
        Returns:
            List[str]: A list of all registered topic names.
        """
        return list(self._narratives.keys())
    
    def update_belief(
        self,
        persona_id: str,
        topic: str,
        position: float,
        confidence: float,
        source_id: Optional[str] = None,
    ) -> None:
        """
        Record a belief update.
        
        Args:
            persona_id: Persona whose belief changed
            topic: Topic of the belief
            position: New position (-1 to 1)
            confidence: Confidence in belief (0 to 1)
            source_id: Persona who influenced this change (if any)
        """
        self.register_topic(topic)
        
        self._persona_beliefs[persona_id][topic] = position
        
        if source_id:
            self._narratives[topic].record_propagation(
                source_id, persona_id, datetime.now()
            )
    
    def take_snapshot(self, topic: str, timestamp: Optional[datetime] = None) -> BeliefSnapshot:
        """Take a snapshot of current belief states for a topic.
        
        Args:
            topic: The topic to snapshot.
            timestamp: Optional timestamp for the snapshot. Defaults to now.
        
        Returns:
            BeliefSnapshot: The created snapshot containing current belief positions.
        """
        timestamp = timestamp or datetime.now()
        
        positions = {}
        confidences = {}
        
        for persona_id, beliefs in self._persona_beliefs.items():
            if topic in beliefs:
                positions[persona_id] = beliefs[topic]
                confidences[persona_id] = 0.5  # Default confidence
        
        snapshot = BeliefSnapshot(
            timestamp=timestamp,
            topic=topic,
            belief_positions=positions,
            belief_confidences=confidences,
        )
        
        if topic in self._narratives:
            self._narratives[topic].add_snapshot(snapshot)
        
        return snapshot
    
    def take_all_snapshots(self, timestamp: Optional[datetime] = None) -> List[BeliefSnapshot]:
        """Take snapshots for all tracked topics.
        
        Args:
            timestamp: Optional timestamp for all snapshots. Defaults to now.
        
        Returns:
            List[BeliefSnapshot]: A list of snapshots, one for each tracked topic.
        """
        return [self.take_snapshot(topic, timestamp) for topic in self._narratives.keys()]
    
    def on_step(self, timestamp: Optional[datetime] = None) -> None:
        """Process a simulation step, potentially taking snapshots.
        
        Automatically takes snapshots at the configured interval.
        
        Args:
            timestamp: Optional timestamp for snapshots. Defaults to now.
        """
        self._step_count += 1
        
        if self._step_count % self._snapshot_interval == 0:
            self.take_all_snapshots(timestamp)
    
    def compute_pairwise_similarity(self, persona_a: str, persona_b: str) -> float:
        """Compute belief similarity between two personas across all topics.
        
        Args:
            persona_a: First persona ID.
            persona_b: Second persona ID.
        
        Returns:
            float: Similarity score from 0 (opposite beliefs) to 1 (identical beliefs).
                Returns 0.5 if personas share no common topics.
        """
        beliefs_a = self._persona_beliefs.get(persona_a, {})
        beliefs_b = self._persona_beliefs.get(persona_b, {})
        
        common_topics = set(beliefs_a.keys()) & set(beliefs_b.keys())
        
        if not common_topics:
            return 0.5  # No common topics, assume neutral similarity
        
        total_diff = sum(
            abs(beliefs_a[topic] - beliefs_b[topic])
            for topic in common_topics
        )
        
        # Max difference per topic is 2 (from -1 to 1)
        max_diff = len(common_topics) * 2
        
        return 1.0 - (total_diff / max_diff)
    
    def compute_similarity_matrix(self, persona_ids: List[str]) -> Dict[Tuple[str, str], float]:
        """Compute pairwise belief similarity for all persona pairs.
        
        Args:
            persona_ids: List of persona IDs to compare.
        
        Returns:
            Dict[Tuple[str, str], float]: A dictionary mapping persona pairs to
                their similarity scores. Both (a, b) and (b, a) keys are included.
        """
        matrix = {}
        
        for i, id_a in enumerate(persona_ids):
            for id_b in persona_ids[i + 1:]:
                similarity = self.compute_pairwise_similarity(id_a, id_b)
                matrix[(id_a, id_b)] = similarity
                matrix[(id_b, id_a)] = similarity
        
        return matrix
    
    def get_narrative_summary(self, topic: str) -> Dict[str, any]:
        """Get a summary of a narrative's evolution.
        
        Args:
            topic: The topic name to summarize.
        
        Returns:
            Dict[str, any]: A dictionary containing narrative statistics including
                consensus score, convergence rate, influential personas, and more.
                Returns an error dict if the topic is not tracked.
        """
        narrative = self._narratives.get(topic)
        if not narrative:
            return {"error": f"Topic '{topic}' not tracked"}
        
        latest = narrative.latest_snapshot
        
        return {
            "topic": topic,
            "created_at": narrative.created_at.isoformat(),
            "interaction_count": narrative.interaction_count,
            "snapshot_count": len(narrative.snapshots),
            "current_consensus": latest.consensus_score if latest else 0.0,
            "current_mean_position": latest.mean_position if latest else 0.0,
            "convergence_rate": narrative.get_convergence_rate(),
            "influential_personas": narrative.get_influential_personas(),
            "cluster_count": len(latest.get_clusters()) if latest else 0,
        }
    
    def get_all_summaries(self) -> List[Dict[str, any]]:
        """Get summaries for all tracked narratives.
        
        Returns:
            List[Dict[str, any]]: A list of summary dictionaries, one for each
                tracked topic.
        """
        return [self.get_narrative_summary(topic) for topic in self._narratives.keys()]
    
    def detect_echo_chambers(
        self,
        similarity_threshold: float = 0.8,
        min_size: int = 3,
    ) -> List[Set[str]]:
        """Detect potential echo chambers based on belief similarity.
        
        Identifies connected groups of personas whose beliefs are highly similar
        using graph-based clustering.
        
        Args:
            similarity_threshold: Minimum similarity score for personas to be
                considered part of the same echo chamber. Defaults to 0.8.
            min_size: Minimum number of personas required to form an echo chamber.
                Defaults to 3.
        
        Returns:
            List[Set[str]]: A list of sets, where each set contains persona IDs
                belonging to the same echo chamber.
        """
        persona_ids = list(self._persona_beliefs.keys())
        
        if len(persona_ids) < min_size:
            return []
        
        # Build similarity graph
        similar_pairs: Dict[str, Set[str]] = defaultdict(set)
        
        for i, id_a in enumerate(persona_ids):
            for id_b in persona_ids[i + 1:]:
                similarity = self.compute_pairwise_similarity(id_a, id_b)
                if similarity >= similarity_threshold:
                    similar_pairs[id_a].add(id_b)
                    similar_pairs[id_b].add(id_a)
        
        # Find connected components
        chambers = []
        visited = set()
        
        for persona_id in persona_ids:
            if persona_id in visited:
                continue
            
            if persona_id not in similar_pairs:
                continue
            
            # BFS to find connected component
            chamber = set()
            queue = [persona_id]
            
            while queue:
                current = queue.pop(0)
                if current in chamber:
                    continue
                
                chamber.add(current)
                visited.add(current)
                
                for neighbor in similar_pairs.get(current, []):
                    if neighbor not in chamber:
                        queue.append(neighbor)
            
            if len(chamber) >= min_size:
                chambers.append(chamber)
        
        return chambers
    
    def clear(self) -> None:
        """Clear all tracking data and reset the step counter."""
        self._narratives.clear()
        self._persona_beliefs.clear()
        self._step_count = 0
    
    def __repr__(self) -> str:
        """Return a string representation of the NarrativeTracker.
        
        Returns:
            str: A string showing the number of tracked topics and personas.
        """
        return f"NarrativeTracker(topics={len(self._narratives)}, personas={len(self._persona_beliefs)})"
