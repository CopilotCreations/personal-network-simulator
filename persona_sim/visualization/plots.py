"""
Plotting utilities for simulation visualization.

Generates visualizations for:
- Narrative convergence graphs
- Persona similarity matrices
- Network topology
- Coordination signals
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json


class SimulationPlotter:
    """
    Creates visualizations for simulation results.
    
    Note: This module provides data structures suitable for plotting
    with matplotlib or other visualization libraries. The actual
    plotting requires matplotlib to be installed.
    """
    
    def __init__(self, output_dir: str = "."):
        """Initialize the simulation plotter.

        Args:
            output_dir: Directory path for saving output files. Defaults to
                current directory.
        """
        self.output_dir = output_dir
        self._has_matplotlib = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available.

        Returns:
            True if matplotlib can be imported, False otherwise.
        """
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def plot_convergence(
        self,
        topic: str,
        timestamps: List[datetime],
        variances: List[float],
        mean_positions: List[float],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot belief convergence over time.

        Args:
            topic: The topic or narrative being tracked.
            timestamps: List of datetime objects for each data point.
            variances: List of belief variance values at each timestamp.
            mean_positions: List of mean belief positions at each timestamp.
            save_path: Optional file path to save the plot image.

        Returns:
            The matplotlib figure if matplotlib is available, otherwise
            returns the data as a dict for external plotting.
        """
        data = {
            "topic": topic,
            "timestamps": [t.isoformat() for t in timestamps],
            "variances": variances,
            "mean_positions": mean_positions,
        }
        
        if not self._has_matplotlib:
            return data
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Variance plot
        ax1.plot(range(len(variances)), variances, 'b-', linewidth=2)
        ax1.set_ylabel('Belief Variance')
        ax1.set_title(f'Narrative Convergence: {topic}')
        ax1.axhline(y=0.2, color='r', linestyle='--', label='Consensus threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mean position plot
        ax2.plot(range(len(mean_positions)), mean_positions, 'g-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Mean Belief Position')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_similarity_matrix(
        self,
        persona_ids: List[str],
        similarity_matrix: Dict[Tuple[str, str], float],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot persona similarity matrix as a heatmap.

        Args:
            persona_ids: List of persona identifier strings.
            similarity_matrix: Dictionary mapping persona ID pairs to similarity
                scores between 0 and 1.
            save_path: Optional file path to save the plot image.

        Returns:
            The matplotlib figure if matplotlib is available, otherwise
            returns the data as a dict for external plotting.
        """
        n = len(persona_ids)
        
        # Build matrix
        matrix_data = [[0.0] * n for _ in range(n)]
        for i, id_a in enumerate(persona_ids):
            matrix_data[i][i] = 1.0
            for j, id_b in enumerate(persona_ids):
                if i != j:
                    key = (id_a, id_b)
                    matrix_data[i][j] = similarity_matrix.get(key, 0.5)
        
        data = {
            "persona_ids": persona_ids,
            "matrix": matrix_data,
        }
        
        if not self._has_matplotlib:
            return data
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix_data, cmap=cmap, vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        
        # Truncate long IDs
        short_ids = [pid[:8] for pid in persona_ids]
        ax.set_xticklabels(short_ids, rotation=45, ha='right')
        ax.set_yticklabels(short_ids)
        
        ax.set_title('Persona Belief Similarity Matrix')
        
        # Colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Similarity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_network_topology(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        coordinated_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot network topology with optional coordination highlighting.

        Args:
            nodes: List of persona IDs representing network nodes.
            edges: List of (source, target, strength) tuples where strength
                is a float between 0 and 1.
            coordinated_ids: Optional list of coordinated persona IDs to
                highlight in red.
            save_path: Optional file path to save the plot image.

        Returns:
            The matplotlib figure if matplotlib is available, otherwise
            returns the data as a dict for external plotting.
        """
        data = {
            "nodes": nodes,
            "edges": [(s, t, w) for s, t, w in edges],
            "coordinated": coordinated_ids or [],
        }
        
        if not self._has_matplotlib:
            return data
        
        try:
            import matplotlib.pyplot as plt
            import math
        except ImportError:
            return data
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Simple circular layout
        n = len(nodes)
        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            positions[node] = (math.cos(angle), math.sin(angle))
        
        # Draw edges
        for source, target, strength in edges:
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                alpha = 0.2 + 0.6 * strength
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=alpha, linewidth=strength * 2)
        
        # Draw nodes
        coordinated_set = set(coordinated_ids or [])
        
        for node, (x, y) in positions.items():
            color = 'red' if node in coordinated_set else 'blue'
            ax.scatter(x, y, c=color, s=200, zorder=5)
            ax.annotate(
                node[:6], (x, y),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8,
            )
        
        ax.set_title('Persona Network Topology')
        ax.axis('equal')
        ax.axis('off')
        
        # Legend
        ax.scatter([], [], c='blue', s=100, label='Normal')
        ax.scatter([], [], c='red', s=100, label='Coordinated')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_coordination_signals(
        self,
        signals: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot coordination signal analysis.

        Args:
            signals: List of signal dictionaries containing 'signal_type'
                and 'strength' keys.
            save_path: Optional file path to save the plot image.

        Returns:
            The matplotlib figure if matplotlib is available, otherwise
            returns the data as a dict for external plotting. Returns an
            error dict if no signals are provided.
        """
        if not signals:
            return {"error": "No signals to plot"}
        
        # Aggregate by type
        type_counts = {}
        type_strengths = {}
        
        for signal in signals:
            sig_type = signal.get("signal_type", "unknown")
            strength = signal.get("strength", 0.5)
            
            type_counts[sig_type] = type_counts.get(sig_type, 0) + 1
            if sig_type not in type_strengths:
                type_strengths[sig_type] = []
            type_strengths[sig_type].append(strength)
        
        data = {
            "type_counts": type_counts,
            "type_avg_strengths": {
                t: sum(s) / len(s) for t, s in type_strengths.items()
            },
        }
        
        if not self._has_matplotlib:
            return data
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal counts by type
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        
        ax1.bar(types, counts, color='steelblue')
        ax1.set_ylabel('Count')
        ax1.set_title('Coordination Signals by Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average strength by type
        avg_strengths = [sum(type_strengths[t]) / len(type_strengths[t]) for t in types]
        colors = ['red' if s > 0.7 else 'orange' if s > 0.4 else 'green' for s in avg_strengths]
        
        ax2.bar(types, avg_strengths, color=colors)
        ax2.set_ylabel('Average Strength')
        ax2.set_title('Signal Strength by Type')
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Strong threshold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_timeline(
        self,
        timeline: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot simulation timeline metrics.

        Args:
            timeline: List of step metrics dicts containing 'step',
                'interactions', and 'belief_changes' keys.
            save_path: Optional file path to save the plot image.

        Returns:
            The matplotlib figure if matplotlib is available, otherwise
            returns the data as a dict for external plotting. Returns an
            error dict if no timeline data is provided.
        """
        if not timeline:
            return {"error": "No timeline data"}
        
        steps = [t["step"] for t in timeline]
        interactions = [t.get("interactions", 0) for t in timeline]
        belief_changes = [t.get("belief_changes", 0) for t in timeline]
        
        data = {
            "steps": steps,
            "interactions": interactions,
            "belief_changes": belief_changes,
        }
        
        if not self._has_matplotlib:
            return data
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(steps, interactions, 'b-', label='Interactions', linewidth=2)
        ax.plot(steps, belief_changes, 'r-', label='Belief Changes', linewidth=2)
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Count')
        ax.set_title('Simulation Activity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def export_plot_data(
        self,
        data: Dict[str, Any],
        filepath: str,
    ) -> None:
        """Export plot data to JSON for external visualization.

        Args:
            data: Dictionary containing plot data to export.
            filepath: Path to the output JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_summary_report(
        self,
        metrics: Dict[str, Any],
        convergence_data: List[Dict[str, Any]],
        coordination_summary: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """Create a text summary report of simulation results.

        Args:
            metrics: Dictionary containing simulation metrics such as
                simulation_id, total_steps, persona_count, etc.
            convergence_data: List of dictionaries with topic convergence
                information including 'topic' and 'convergence_rate'.
            coordination_summary: Dictionary containing coordination detection
                results including total_signals, strong_signals, and
                coordinated_personas.
            save_path: Optional file path to save the text report.

        Returns:
            The formatted report as a string.
        """
        lines = [
            "=" * 60,
            "PERSONA NETWORK SIMULATION REPORT",
            "=" * 60,
            "",
            "SIMULATION OVERVIEW",
            "-" * 40,
            f"Simulation ID: {metrics.get('simulation_id', 'N/A')}",
            f"Total Steps: {metrics.get('total_steps', 0)}",
            f"Total Interactions: {metrics.get('total_interactions', 0)}",
            f"Belief Changes: {metrics.get('total_belief_changes', 0)}",
            "",
            "NETWORK STATISTICS",
            "-" * 40,
            f"Personas: {metrics.get('persona_count', 0)}",
            f"Coordinated Personas: {metrics.get('coordinated_persona_count', 0)}",
            f"Connections: {metrics.get('connection_count', 0)}",
            f"Graph Density: {metrics.get('graph_density', 0):.4f}",
            f"Avg Clustering: {metrics.get('avg_clustering', 0):.4f}",
            "",
            "NARRATIVE CONVERGENCE",
            "-" * 40,
            f"Topics Tracked: {metrics.get('topics_tracked', 0)}",
            f"Topics with Consensus: {metrics.get('topics_with_consensus', 0)}",
            f"Avg Convergence Rate: {metrics.get('avg_convergence_rate', 0):.4f}",
            "",
        ]
        
        if convergence_data:
            lines.append("Top Converging Topics:")
            for topic_data in convergence_data[:5]:
                lines.append(f"  - {topic_data.get('topic', 'N/A')}: "
                           f"rate={topic_data.get('convergence_rate', 0):.4f}")
        
        lines.extend([
            "",
            "COORDINATION DETECTION",
            "-" * 40,
            f"Total Signals: {coordination_summary.get('total_signals', 0)}",
            f"Strong Signals: {coordination_summary.get('strong_signals', 0)}",
            "",
        ])
        
        signals_by_type = coordination_summary.get('signals_by_type', {})
        if signals_by_type:
            lines.append("Signals by Type:")
            for sig_type, count in signals_by_type.items():
                lines.append(f"  - {sig_type}: {count}")
        
        coordinated = coordination_summary.get('coordinated_personas', [])
        if coordinated:
            lines.extend([
                "",
                "Highly Coordinated Personas:",
            ])
            for persona_id, signal_count in coordinated[:5]:
                lines.append(f"  - {persona_id[:8]}...: {signal_count} signals")
        
        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def __repr__(self) -> str:
        """Return string representation of the plotter.

        Returns:
            String indicating matplotlib availability status.
        """
        return f"SimulationPlotter(matplotlib={'available' if self._has_matplotlib else 'not available'})"
