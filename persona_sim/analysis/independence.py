"""
Independence detection for coordination analysis.

Detects signals that indicate coordinated (non-independent) behavior
among personas, useful for identifying synthetic persona farms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import math

from ..network.dynamics import InteractionEvent, InteractionScheduler
from ..agents.style import LinguisticStyle


@dataclass
class CoordinationSignal:
    """
    A detected signal of potential coordination.
    
    Includes type, strength, and involved personas.
    """
    signal_type: str  # "timing", "phrasing", "belief_sync", "response_pattern"
    strength: float  # 0.0 to 1.0
    persona_ids: List[str]
    topic: Optional[str]
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_strong(self) -> bool:
        """Whether this is a strong coordination signal."""
        return self.strength > 0.7
    
    @property
    def is_moderate(self) -> bool:
        """Whether this is a moderate coordination signal."""
        return 0.4 < self.strength <= 0.7


class IndependenceDetector:
    """
    Detects coordination signals in persona network behavior.
    
    Analyzes:
    - Timing correlation (synchronized posting)
    - Phrasing similarity (similar language patterns)
    - Belief update synchronicity (beliefs changing together)
    - Response patterns (always responding to same sources)
    """
    
    def __init__(
        self,
        timing_window_minutes: float = 5.0,
        similarity_threshold: float = 0.7,
    ):
        """
        Args:
            timing_window_minutes: Window for considering events synchronized
            similarity_threshold: Threshold for considering content similar
        """
        self.timing_window = timedelta(minutes=timing_window_minutes)
        self.similarity_threshold = similarity_threshold
        self._signals: List[CoordinationSignal] = []
    
    def analyze_timing_correlation(
        self,
        events: List[InteractionEvent],
        persona_ids: List[str],
    ) -> List[CoordinationSignal]:
        """Detect timing-based coordination signals.

        Looks for personas that post within suspiciously short
        time windows of each other.

        Args:
            events: List of interaction events to analyze.
            persona_ids: List of persona IDs to check for coordination.

        Returns:
            List of coordination signals detected from timing patterns.
        """
        signals = []
        
        # Group events by topic
        events_by_topic: Dict[str, List[InteractionEvent]] = defaultdict(list)
        for event in events:
            events_by_topic[event.topic].append(event)
        
        for topic, topic_events in events_by_topic.items():
            # Sort by time
            sorted_events = sorted(topic_events, key=lambda e: e.timestamp)
            
            # Find clusters of near-simultaneous events
            clusters = self._find_timing_clusters(sorted_events)
            
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                
                # Check if this cluster involves our target personas
                cluster_personas = set(e.source_id for e in cluster)
                target_in_cluster = cluster_personas & set(persona_ids)
                
                if len(target_in_cluster) >= 2:
                    # Calculate synchronization strength
                    time_spread = (
                        cluster[-1].timestamp - cluster[0].timestamp
                    ).total_seconds() / 60
                    
                    # Smaller spread = higher strength
                    strength = max(0.0, 1.0 - time_spread / self.timing_window.seconds * 60)
                    
                    if strength > 0.3:
                        signal = CoordinationSignal(
                            signal_type="timing",
                            strength=strength,
                            persona_ids=list(target_in_cluster),
                            topic=topic,
                            description=f"{len(target_in_cluster)} personas posted within {time_spread:.1f} minutes on '{topic}'",
                        )
                        signals.append(signal)
        
        self._signals.extend(signals)
        return signals
    
    def _find_timing_clusters(
        self,
        sorted_events: List[InteractionEvent],
    ) -> List[List[InteractionEvent]]:
        """Find clusters of events that occur close together.

        Args:
            sorted_events: List of interaction events sorted by timestamp.

        Returns:
            List of clusters, where each cluster is a list of events that
            occurred within the timing window of each other.
        """
        if not sorted_events:
            return []
        
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            time_diff = event.timestamp - current_cluster[-1].timestamp
            
            if time_diff <= self.timing_window:
                current_cluster.append(event)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [event]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def analyze_phrasing_similarity(
        self,
        events: List[InteractionEvent],
        styles: Dict[str, LinguisticStyle],
    ) -> List[CoordinationSignal]:
        """Detect phrasing-based coordination signals.

        Looks for suspiciously similar language patterns across
        personas' event content and linguistic styles.

        Args:
            events: List of interaction events to analyze for content similarity.
            styles: Dictionary mapping persona IDs to their linguistic styles.

        Returns:
            List of coordination signals detected from phrasing patterns.
        """
        signals = []
        
        # Group events by topic
        events_by_topic: Dict[str, List[InteractionEvent]] = defaultdict(list)
        for event in events:
            if event.content:
                events_by_topic[event.topic].append(event)
        
        for topic, topic_events in events_by_topic.items():
            # Compare all pairs
            similar_groups = self._find_similar_content_groups(topic_events)
            
            for group in similar_groups:
                persona_ids = list(set(e.source_id for e in group))
                
                if len(persona_ids) >= 2:
                    signal = CoordinationSignal(
                        signal_type="phrasing",
                        strength=0.8,  # High similarity = high strength
                        persona_ids=persona_ids,
                        topic=topic,
                        description=f"{len(persona_ids)} personas used very similar phrasing on '{topic}'",
                    )
                    signals.append(signal)
        
        # Also check style similarity
        style_signals = self._analyze_style_similarity(styles)
        signals.extend(style_signals)
        
        self._signals.extend(signals)
        return signals
    
    def _find_similar_content_groups(
        self,
        events: List[InteractionEvent],
    ) -> List[List[InteractionEvent]]:
        """Find groups of events with similar content.

        Args:
            events: List of interaction events to compare.

        Returns:
            List of groups, where each group contains events with
            content similarity above the threshold.
        """
        if len(events) < 2:
            return []
        
        groups = []
        used = set()
        
        for i, event_a in enumerate(events):
            if i in used:
                continue
            
            group = [event_a]
            
            for j, event_b in enumerate(events[i + 1:], i + 1):
                if j in used:
                    continue
                
                similarity = self._compute_content_similarity(
                    event_a.content, event_b.content
                )
                
                if similarity >= self.similarity_threshold:
                    group.append(event_b)
                    used.add(j)
            
            if len(group) >= 2:
                groups.append(group)
                used.add(i)
        
        return groups
    
    def _compute_content_similarity(self, text_a: str, text_b: str) -> float:
        """Compute simple similarity between two texts.

        Uses word overlap (Jaccard similarity) to measure how similar
        two text strings are.

        Args:
            text_a: First text string to compare.
            text_b: Second text string to compare.

        Returns:
            Similarity score between 0.0 (no overlap) and 1.0 (identical words).
        """
        if not text_a or not text_b:
            return 0.0
        
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_style_similarity(
        self,
        styles: Dict[str, LinguisticStyle],
    ) -> List[CoordinationSignal]:
        """Find personas with suspiciously similar linguistic styles.

        Args:
            styles: Dictionary mapping persona IDs to their linguistic styles.

        Returns:
            List of coordination signals for personas with similar styles.
        """
        signals = []
        persona_ids = list(styles.keys())
        
        similar_pairs = []
        
        for i, id_a in enumerate(persona_ids):
            for id_b in persona_ids[i + 1:]:
                similarity = LinguisticStyle.compute_similarity(
                    styles[id_a], styles[id_b]
                )
                
                if similarity >= self.similarity_threshold:
                    similar_pairs.append((id_a, id_b, similarity))
        
        # Group similar pairs into clusters
        if similar_pairs:
            # Simple clustering: merge pairs that share members
            clusters: List[Set[str]] = []
            
            for id_a, id_b, sim in similar_pairs:
                merged = False
                for cluster in clusters:
                    if id_a in cluster or id_b in cluster:
                        cluster.add(id_a)
                        cluster.add(id_b)
                        merged = True
                        break
                
                if not merged:
                    clusters.append({id_a, id_b})
            
            for cluster in clusters:
                if len(cluster) >= 2:
                    signal = CoordinationSignal(
                        signal_type="phrasing",
                        strength=0.7,
                        persona_ids=list(cluster),
                        topic=None,
                        description=f"{len(cluster)} personas have very similar linguistic styles",
                    )
                    signals.append(signal)
        
        return signals
    
    def analyze_belief_synchronicity(
        self,
        belief_history: Dict[str, List[Tuple[datetime, str, float]]],
    ) -> List[CoordinationSignal]:
        """Detect belief updates that happen synchronously.

        Identifies personas that update their beliefs on the same topics
        at similar times and to similar positions.

        Args:
            belief_history: Dictionary mapping persona_id to list of
                (timestamp, topic, position) tuples representing belief changes.

        Returns:
            List of coordination signals for synchronized belief updates.
        """
        signals = []
        
        # Group belief changes by topic
        changes_by_topic: Dict[str, List[Tuple[str, datetime, float]]] = defaultdict(list)
        
        for persona_id, history in belief_history.items():
            for timestamp, topic, position in history:
                changes_by_topic[topic].append((persona_id, timestamp, position))
        
        for topic, changes in changes_by_topic.items():
            # Sort by time
            sorted_changes = sorted(changes, key=lambda x: x[1])
            
            # Find synchronized changes
            sync_clusters = self._find_sync_belief_changes(sorted_changes)
            
            for cluster in sync_clusters:
                persona_ids = [c[0] for c in cluster]
                positions = [c[2] for c in cluster]
                
                # Check if positions are similar (moving in same direction)
                avg_position = sum(positions) / len(positions)
                position_variance = sum((p - avg_position) ** 2 for p in positions) / len(positions)
                
                if position_variance < 0.1:  # Similar positions
                    time_spread = (cluster[-1][1] - cluster[0][1]).total_seconds() / 60
                    
                    signal = CoordinationSignal(
                        signal_type="belief_sync",
                        strength=0.8 if position_variance < 0.05 else 0.6,
                        persona_ids=persona_ids,
                        topic=topic,
                        description=f"{len(persona_ids)} personas updated beliefs on '{topic}' within {time_spread:.1f} minutes to similar positions",
                    )
                    signals.append(signal)
        
        self._signals.extend(signals)
        return signals
    
    def _find_sync_belief_changes(
        self,
        sorted_changes: List[Tuple[str, datetime, float]],
    ) -> List[List[Tuple[str, datetime, float]]]:
        """Find clusters of synchronized belief changes.

        Args:
            sorted_changes: List of (persona_id, timestamp, position) tuples
                sorted by timestamp.

        Returns:
            List of clusters, where each cluster contains belief changes
            that occurred within the timing window of each other.
        """
        if len(sorted_changes) < 2:
            return []
        
        clusters = []
        current_cluster = [sorted_changes[0]]
        
        for change in sorted_changes[1:]:
            time_diff = change[1] - current_cluster[-1][1]
            
            if time_diff <= self.timing_window:
                current_cluster.append(change)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [change]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def analyze_response_patterns(
        self,
        events: List[InteractionEvent],
    ) -> List[CoordinationSignal]:
        """Detect suspicious response patterns.

        Looks for personas that consistently respond to the same sources,
        which may indicate coordinated behavior.

        Args:
            events: List of interaction events to analyze.

        Returns:
            List of coordination signals for suspicious response patterns.
        """
        signals = []
        
        # Track who responds to whom
        response_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            if event.target_id:
                response_matrix[event.source_id][event.target_id] += 1
        
        # Find personas with identical response patterns
        persona_patterns: Dict[str, Set[str]] = {}
        
        for responder, targets in response_matrix.items():
            # Get primary targets (responded to >= 3 times)
            primary_targets = frozenset(
                t for t, count in targets.items() if count >= 3
            )
            if primary_targets:
                persona_patterns[responder] = primary_targets
        
        # Group personas with identical patterns
        pattern_groups: Dict[frozenset, List[str]] = defaultdict(list)
        
        for persona_id, pattern in persona_patterns.items():
            pattern_groups[frozenset(pattern)].append(persona_id)
        
        for pattern, persona_ids in pattern_groups.items():
            if len(persona_ids) >= 2:
                signal = CoordinationSignal(
                    signal_type="response_pattern",
                    strength=0.7,
                    persona_ids=persona_ids,
                    topic=None,
                    description=f"{len(persona_ids)} personas consistently respond to the same {len(pattern)} sources",
                )
                signals.append(signal)
        
        self._signals.extend(signals)
        return signals
    
    def compute_independence_score(
        self,
        persona_id: str,
        all_signals: Optional[List[CoordinationSignal]] = None,
    ) -> float:
        """Compute an independence score for a persona.

        Args:
            persona_id: The ID of the persona to score.
            all_signals: Optional list of signals to use. If None, uses
                internally tracked signals.

        Returns:
            Independence score from 0.0 (highly coordinated) to 1.0
            (fully independent).
        """
        signals = all_signals or self._signals
        
        # Find all signals involving this persona
        involving = [s for s in signals if persona_id in s.persona_ids]
        
        if not involving:
            return 1.0  # No coordination signals = independent
        
        # Weight signals by strength
        weighted_sum = sum(s.strength for s in involving)
        
        # Normalize: more signals = lower independence
        max_expected_signals = 10
        normalized = min(weighted_sum / max_expected_signals, 1.0)
        
        return 1.0 - normalized
    
    def get_all_signals(self) -> List[CoordinationSignal]:
        """Get all detected coordination signals.

        Returns:
            Copy of the list of all coordination signals detected so far.
        """
        return self._signals.copy()
    
    def get_strong_signals(self) -> List[CoordinationSignal]:
        """Get only strong coordination signals.

        Returns:
            List of coordination signals with strength > 0.7.
        """
        return [s for s in self._signals if s.is_strong]
    
    def get_signals_for_persona(self, persona_id: str) -> List[CoordinationSignal]:
        """Get all signals involving a specific persona.

        Args:
            persona_id: The ID of the persona to filter signals for.

        Returns:
            List of coordination signals that include the specified persona.
        """
        return [s for s in self._signals if persona_id in s.persona_ids]
    
    def get_coordination_summary(self) -> Dict[str, any]:
        """Get a summary of detected coordination.

        Returns:
            Dictionary containing:
                - total_signals: Total number of coordination signals.
                - strong_signals: Number of signals with strength > 0.7.
                - moderate_signals: Number of signals with 0.4 < strength <= 0.7.
                - signals_by_type: Count of signals by type.
                - coordinated_personas: Top 10 personas by signal involvement.
        """
        signals = self._signals
        
        if not signals:
            return {
                "total_signals": 0,
                "strong_signals": 0,
                "coordinated_personas": [],
            }
        
        # Find personas involved in multiple signals
        persona_signal_counts: Dict[str, int] = defaultdict(int)
        for signal in signals:
            for persona_id in signal.persona_ids:
                persona_signal_counts[persona_id] += 1
        
        # Personas in 3+ signals are likely coordinated
        highly_coordinated = [
            (pid, count) for pid, count in persona_signal_counts.items()
            if count >= 3
        ]
        highly_coordinated.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_signals": len(signals),
            "strong_signals": len([s for s in signals if s.is_strong]),
            "moderate_signals": len([s for s in signals if s.is_moderate]),
            "signals_by_type": {
                "timing": len([s for s in signals if s.signal_type == "timing"]),
                "phrasing": len([s for s in signals if s.signal_type == "phrasing"]),
                "belief_sync": len([s for s in signals if s.signal_type == "belief_sync"]),
                "response_pattern": len([s for s in signals if s.signal_type == "response_pattern"]),
            },
            "coordinated_personas": highly_coordinated[:10],
        }
    
    def clear_signals(self) -> None:
        """Clear all detected signals.

        Removes all coordination signals from internal storage.
        """
        self._signals.clear()
    
    def __repr__(self) -> str:
        """Return string representation of the detector.

        Returns:
            String showing the number of signals currently tracked.
        """
        return f"IndependenceDetector(signals={len(self._signals)})"
