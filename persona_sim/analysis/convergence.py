"""
Convergence analysis for narrative tracking.

Measures how beliefs and opinions converge across the persona network,
identifying patterns of consensus formation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math

from ..simulation.narratives import NarrativeTracker, Narrative, BeliefSnapshot


@dataclass
class ConvergenceMetrics:
    """Metrics describing belief convergence for a topic."""
    topic: str
    initial_variance: float
    final_variance: float
    convergence_rate: float  # Per hour
    time_to_half_convergence: Optional[float]  # Hours, if converging
    consensus_reached: bool
    consensus_position: Optional[float]
    polarization_index: float  # 0 = no polarization, 1 = fully polarized
    cluster_count: int
    
    @property
    def variance_reduction(self) -> float:
        """Percentage reduction in variance."""
        if self.initial_variance == 0:
            return 0.0
        return (self.initial_variance - self.final_variance) / self.initial_variance
    
    @property
    def is_converging(self) -> bool:
        """Whether beliefs are converging."""
        return self.convergence_rate > 0


class ConvergenceAnalyzer:
    """
    Analyzes convergence patterns in belief evolution.
    
    Detects:
    - Consensus formation
    - Polarization (convergence to multiple poles)
    - Rate of convergence
    - Artificially rapid convergence (coordination signal)
    """
    
    def __init__(self, consensus_threshold: float = 0.2):
        """
        Args:
            consensus_threshold: Maximum variance for consensus (default 0.2)
        """
        self.consensus_threshold = consensus_threshold
    
    def analyze_narrative(self, narrative: Narrative) -> ConvergenceMetrics:
        """Analyze convergence for a single narrative."""
        if not narrative.snapshots:
            return self._empty_metrics(narrative.topic)
        
        first = narrative.snapshots[0]
        last = narrative.snapshots[-1]
        
        initial_var = first.position_variance
        final_var = last.position_variance
        
        # Calculate convergence rate
        if len(narrative.snapshots) >= 2:
            time_delta = (last.timestamp - first.timestamp).total_seconds() / 3600
            if time_delta > 0:
                convergence_rate = (initial_var - final_var) / time_delta
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Check for consensus
        consensus_reached = final_var <= self.consensus_threshold
        consensus_position = last.mean_position if consensus_reached else None
        
        # Calculate time to half convergence
        time_to_half = self._calculate_half_convergence_time(narrative.snapshots)
        
        # Calculate polarization
        polarization = self._calculate_polarization(last)
        
        # Count clusters
        clusters = last.get_clusters(threshold=0.3)
        
        return ConvergenceMetrics(
            topic=narrative.topic,
            initial_variance=initial_var,
            final_variance=final_var,
            convergence_rate=convergence_rate,
            time_to_half_convergence=time_to_half,
            consensus_reached=consensus_reached,
            consensus_position=consensus_position,
            polarization_index=polarization,
            cluster_count=len(clusters),
        )
    
    def _empty_metrics(self, topic: str) -> ConvergenceMetrics:
        """Create empty metrics for topics with no data."""
        return ConvergenceMetrics(
            topic=topic,
            initial_variance=0.0,
            final_variance=0.0,
            convergence_rate=0.0,
            time_to_half_convergence=None,
            consensus_reached=False,
            consensus_position=None,
            polarization_index=0.0,
            cluster_count=0,
        )
    
    def _calculate_half_convergence_time(
        self,
        snapshots: List[BeliefSnapshot],
    ) -> Optional[float]:
        """Calculate time to reach half of initial variance."""
        if len(snapshots) < 2:
            return None
        
        initial_var = snapshots[0].position_variance
        half_var = initial_var / 2
        
        for snapshot in snapshots[1:]:
            if snapshot.position_variance <= half_var:
                delta = (snapshot.timestamp - snapshots[0].timestamp).total_seconds()
                return delta / 3600  # Convert to hours
        
        return None
    
    def _calculate_polarization(self, snapshot: BeliefSnapshot) -> float:
        """
        Calculate polarization index.
        
        High polarization = beliefs clustered at extreme positions.
        """
        if not snapshot.belief_positions:
            return 0.0
        
        positions = list(snapshot.belief_positions.values())
        
        # Count personas at extremes
        extreme_positive = sum(1 for p in positions if p > 0.5)
        extreme_negative = sum(1 for p in positions if p < -0.5)
        moderate = len(positions) - extreme_positive - extreme_negative
        
        total = len(positions)
        
        # Polarization is high when positions are bimodal at extremes
        if extreme_positive == 0 or extreme_negative == 0:
            return 0.0  # Not polarized, just consensus
        
        # Measure balance between extremes and absence of moderates
        balance = 1.0 - abs(extreme_positive - extreme_negative) / total
        lack_of_moderates = 1.0 - moderate / total
        
        return balance * lack_of_moderates
    
    def analyze_all_narratives(
        self,
        tracker: NarrativeTracker,
    ) -> List[ConvergenceMetrics]:
        """Analyze convergence for all tracked narratives."""
        metrics = []
        
        for topic in tracker.topics:
            narrative = tracker.get_narrative(topic)
            if narrative:
                metrics.append(self.analyze_narrative(narrative))
        
        return metrics
    
    def detect_anomalous_convergence(
        self,
        metrics: List[ConvergenceMetrics],
        rate_threshold: float = 0.5,
    ) -> List[ConvergenceMetrics]:
        """
        Detect narratives with anomalously fast convergence.
        
        Fast convergence can indicate coordinated behavior.
        """
        # Calculate baseline rate
        rates = [m.convergence_rate for m in metrics if m.convergence_rate > 0]
        
        if not rates:
            return []
        
        mean_rate = sum(rates) / len(rates)
        std_rate = (sum((r - mean_rate) ** 2 for r in rates) / len(rates)) ** 0.5
        
        threshold = mean_rate + rate_threshold * std_rate
        
        return [m for m in metrics if m.convergence_rate > threshold]
    
    def compare_convergence_curves(
        self,
        narrative_a: Narrative,
        narrative_b: Narrative,
    ) -> float:
        """
        Compare convergence curves of two narratives.
        
        Returns similarity score from 0 (different) to 1 (identical).
        """
        snapshots_a = narrative_a.snapshots
        snapshots_b = narrative_b.snapshots
        
        if not snapshots_a or not snapshots_b:
            return 0.0
        
        # Normalize to same number of points
        min_len = min(len(snapshots_a), len(snapshots_b))
        
        variances_a = [s.position_variance for s in snapshots_a[:min_len]]
        variances_b = [s.position_variance for s in snapshots_b[:min_len]]
        
        # Calculate correlation
        if len(variances_a) < 2:
            return 0.0
        
        mean_a = sum(variances_a) / len(variances_a)
        mean_b = sum(variances_b) / len(variances_b)
        
        numerator = sum(
            (a - mean_a) * (b - mean_b)
            for a, b in zip(variances_a, variances_b)
        )
        
        denom_a = sum((a - mean_a) ** 2 for a in variances_a) ** 0.5
        denom_b = sum((b - mean_b) ** 2 for b in variances_b) ** 0.5
        
        if denom_a == 0 or denom_b == 0:
            return 0.0
        
        correlation = numerator / (denom_a * denom_b)
        
        # Convert to 0-1 similarity
        return (correlation + 1) / 2
    
    def get_convergence_summary(
        self,
        metrics: List[ConvergenceMetrics],
    ) -> Dict[str, any]:
        """Get a summary of convergence across all narratives."""
        if not metrics:
            return {"error": "No metrics available"}
        
        converging = [m for m in metrics if m.is_converging]
        consensus = [m for m in metrics if m.consensus_reached]
        polarized = [m for m in metrics if m.polarization_index > 0.5]
        
        avg_rate = (
            sum(m.convergence_rate for m in converging) / len(converging)
            if converging else 0.0
        )
        
        return {
            "total_narratives": len(metrics),
            "converging_count": len(converging),
            "consensus_count": len(consensus),
            "polarized_count": len(polarized),
            "average_convergence_rate": avg_rate,
            "topics_with_consensus": [m.topic for m in consensus],
            "topics_polarized": [m.topic for m in polarized],
        }
    
    def __repr__(self) -> str:
        return f"ConvergenceAnalyzer(threshold={self.consensus_threshold})"
