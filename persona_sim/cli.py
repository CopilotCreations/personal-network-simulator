"""
Command-line interface for PersonaNetworkSimulator.

Provides commands for running simulations, analyzing results,
and generating reports.
"""

import argparse
import sys
import json
from datetime import datetime
from typing import Optional
import random

from .agents.persona import Persona, PersonaProfile, PoliticalLeaning
from .agents.style import StyleConstraints, ToneLevel, EmotionalTone
from .network.graph import SocialGraph
from .simulation.engine import SimulationEngine, SimulationConfig
from .simulation.narratives import NarrativeTracker
from .analysis.convergence import ConvergenceAnalyzer
from .analysis.independence import IndependenceDetector
from .analysis.metrics import MetricsCollector
from .visualization.plots import SimulationPlotter


def create_sample_personas(n: int, seed: Optional[int] = None) -> list:
    """Create sample personas for demonstration."""
    rng = random.Random(seed)
    
    names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul",
        "Quinn", "Rose", "Sam", "Tina", "Uma", "Vic", "Wendy", "Xander",
    ]
    
    occupations = [
        "Engineer", "Teacher", "Doctor", "Artist", "Lawyer", "Chef",
        "Writer", "Analyst", "Designer", "Manager", "Researcher", "Student",
    ]
    
    interests = [
        "technology", "environment", "economics", "health", "education",
        "politics", "culture", "science", "sports", "music", "travel",
    ]
    
    personas = []
    for i in range(n):
        name = names[i % len(names)] + (str(i // len(names) + 1) if i >= len(names) else "")
        
        profile = PersonaProfile(
            name=name,
            age=rng.randint(22, 65),
            occupation=rng.choice(occupations),
            interests=rng.sample(interests, k=rng.randint(2, 5)),
            political_leaning=rng.choice(list(PoliticalLeaning)),
            credibility_weight=rng.uniform(0.3, 0.9),
            susceptibility=rng.uniform(0.2, 0.8),
        )
        
        personas.append(Persona(profile))
    
    return personas


def create_sample_styles(personas: list, seed: Optional[int] = None) -> dict:
    """Create sample linguistic styles for personas."""
    rng = random.Random(seed)
    
    styles = {}
    for persona in personas:
        constraints = StyleConstraints(
            tone_level=rng.choice(list(ToneLevel)),
            emotional_tone=rng.choice(list(EmotionalTone)),
            avg_sentence_length=rng.randint(10, 25),
            vocabulary_complexity=rng.uniform(0.2, 0.9),
            use_contractions=rng.random() > 0.3,
            use_emoji=rng.random() > 0.8,
            punctuation_style=rng.choice(["minimal", "standard", "expressive"]),
        )
        styles[persona.id] = constraints
    
    return styles


def run_demo_simulation(args):
    """Run a demonstration simulation."""
    print("=" * 60)
    print("PersonaNetworkSimulator - Demo Simulation")
    print("=" * 60)
    print()
    
    seed = args.seed
    rng = random.Random(seed)
    
    # Create personas
    print(f"Creating {args.personas} personas...")
    personas = create_sample_personas(args.personas, seed)
    styles = create_sample_styles(personas, seed)
    
    # Determine coordinated personas
    coordinated_count = int(args.personas * args.coordinated_fraction)
    coordinated_personas = rng.sample(personas, coordinated_count) if coordinated_count > 0 else []
    coordinated_ids = [p.id for p in coordinated_personas]
    
    print(f"  - Total personas: {len(personas)}")
    print(f"  - Coordinated personas: {len(coordinated_personas)}")
    
    # Create simulation config
    config = SimulationConfig(
        duration_hours=args.hours,
        time_step_minutes=15.0,
        interactions_per_step=max(1, args.personas // 5),
        inject_coordination=len(coordinated_personas) > 0,
        coordinated_persona_fraction=args.coordinated_fraction,
        coordination_topics=["topic_alpha", "topic_beta"],
        seed=seed,
    )
    
    # Create and configure engine
    engine = SimulationEngine(config)
    
    # Add personas
    for persona in personas:
        is_coordinated = persona.id in coordinated_ids
        engine.add_persona(
            persona,
            styles.get(persona.id),
            is_coordinated=is_coordinated,
        )
        
        # Add some initial beliefs
        for topic in ["topic_alpha", "topic_beta", "topic_gamma"]:
            position = rng.uniform(-1, 1)
            if is_coordinated and topic in config.coordination_topics:
                # Coordinated personas start with similar positions
                position = 0.7 + rng.uniform(-0.1, 0.1)
            persona.add_belief(topic, position, confidence=rng.uniform(0.3, 0.7))
    
    # Create network
    print("\nCreating social network...")
    graph = SocialGraph.create_small_world(
        [p.id for p in personas],
        k=min(4, len(personas) - 1),
        rewire_prob=0.1,
        seed=seed,
    )
    
    # Copy graph to engine
    for edge in graph.edges:
        engine.add_connection(edge.source_id, edge.target_id, edge.strength)
    
    print(f"  - Connections: {graph.edge_count}")
    print(f"  - Density: {graph.density():.4f}")
    
    # Set up tracking
    tracker = NarrativeTracker()
    for topic in ["topic_alpha", "topic_beta", "topic_gamma"]:
        tracker.register_topic(topic)
    
    metrics_collector = MetricsCollector()
    
    # Register callbacks
    step_interactions = [0]
    
    def on_interaction(event):
        step_interactions[0] += 1
        metrics_collector.record_interaction(event)
        
        # Update narrative tracker
        source = engine.get_persona(event.source_id)
        if source:
            belief = source.get_belief(event.topic)
            if belief:
                tracker.update_belief(
                    event.source_id,
                    event.topic,
                    belief.position,
                    belief.confidence,
                    event.target_id,
                )
    
    def on_step(state):
        metrics_collector.record_step(state, step_interactions[0], 0)
        tracker.on_step(state.current_time)
        step_interactions[0] = 0
    
    engine.on_interaction(on_interaction)
    engine.on_step(on_step)
    
    # Run simulation
    print(f"\nRunning simulation for {args.hours} hours...")
    state = engine.run()
    
    print(f"  - Steps completed: {state.step_count}")
    print(f"  - Total interactions: {state.total_interactions}")
    print(f"  - Belief changes: {state.belief_changes}")
    
    # Analyze results
    print("\nAnalyzing results...")
    
    # Convergence analysis
    convergence_analyzer = ConvergenceAnalyzer()
    convergence_metrics = convergence_analyzer.analyze_all_narratives(tracker)
    convergence_summary = convergence_analyzer.get_convergence_summary(convergence_metrics)
    
    print(f"  - Topics with consensus: {convergence_summary['consensus_count']}")
    print(f"  - Polarized topics: {convergence_summary['polarized_count']}")
    
    # Independence detection
    detector = IndependenceDetector()
    events = engine.get_all_events()
    
    timing_signals = detector.analyze_timing_correlation(events, coordinated_ids)
    phrasing_signals = detector.analyze_phrasing_similarity(
        events, 
        {pid: engine.get_style(pid) for pid in coordinated_ids if engine.get_style(pid)}
    )
    response_signals = detector.analyze_response_patterns(events)
    
    coordination_summary = detector.get_coordination_summary()
    
    print(f"  - Coordination signals detected: {coordination_summary['total_signals']}")
    print(f"  - Strong signals: {coordination_summary['strong_signals']}")
    
    # Generate report
    plotter = SimulationPlotter(args.output_dir)
    
    final_metrics = metrics_collector.finalize(
        persona_count=len(personas),
        coordinated_count=len(coordinated_personas),
        connection_count=engine.graph.edge_count,
        graph_density=engine.graph.density(),
        avg_clustering=engine.graph.average_clustering(),
        topics_tracked=len(tracker.topics),
        topics_with_consensus=convergence_summary['consensus_count'],
        avg_convergence_rate=convergence_summary['average_convergence_rate'],
        coordination_signals=coordination_summary['total_signals'],
        strong_signals=coordination_summary['strong_signals'],
    )
    
    # Create convergence data for report
    convergence_data = [
        {
            "topic": m.topic,
            "convergence_rate": m.convergence_rate,
            "consensus_reached": m.consensus_reached,
        }
        for m in convergence_metrics
    ]
    
    report = plotter.create_summary_report(
        final_metrics.to_dict(),
        convergence_data,
        coordination_summary,
    )
    
    print("\n" + report)
    
    # Save outputs if requested
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save report
        report_path = os.path.join(args.output_dir, "simulation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # Save metrics JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            f.write(final_metrics.to_json())
        print(f"Metrics saved to: {metrics_path}")
    
    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="persona_sim",
        description="""
PersonaNetworkSimulator - Research Framework for Synthetic Persona Detection

A research and simulation framework for studying the emergence of apparent 
consensus from synthetic personas in controlled environments. Designed for 
platform safety research, NOT deployment.

WARNING: This tool is for research purposes only. Do not use to create or 
deploy synthetic persona networks on real platforms.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a demonstration simulation",
    )
    demo_parser.add_argument(
        "-n", "--personas",
        type=int,
        default=20,
        help="Number of personas to create (default: 20)",
    )
    demo_parser.add_argument(
        "-c", "--coordinated-fraction",
        type=float,
        default=0.2,
        help="Fraction of coordinated personas (default: 0.2)",
    )
    demo_parser.add_argument(
        "-t", "--hours",
        type=float,
        default=24.0,
        help="Simulation duration in hours (default: 24)",
    )
    demo_parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    demo_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory for output files",
    )
    
    # Version command
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version information",
    )
    
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"PersonaNetworkSimulator v{__version__}")
        return 0
    
    if args.command == "demo":
        return run_demo_simulation(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
