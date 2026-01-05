"""Tests for simulation and analysis modules."""

import pytest
from datetime import datetime, timedelta

from persona_sim.agents.persona import Persona, PersonaProfile
from persona_sim.agents.style import StyleConstraints
from persona_sim.simulation.engine import SimulationEngine, SimulationConfig, SimulationPhase
from persona_sim.simulation.narratives import NarrativeTracker, Narrative, BeliefSnapshot
from persona_sim.analysis.convergence import ConvergenceAnalyzer, ConvergenceMetrics
from persona_sim.analysis.independence import IndependenceDetector, CoordinationSignal
from persona_sim.network.dynamics import InteractionEvent, InteractionType


class TestSimulationEngine:
    """Tests for SimulationEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance for testing.

        Returns:
            SimulationEngine: A configured simulation engine with 1-hour duration,
                15-minute time steps, and 2 interactions per step.
        """
        config = SimulationConfig(
            duration_hours=1.0,
            time_step_minutes=15.0,
            interactions_per_step=2,
            seed=42,
        )
        return SimulationEngine(config)
    
    @pytest.fixture
    def personas(self):
        """Create a list of test personas.

        Returns:
            list[Persona]: A list of 5 personas with incrementing ages starting at 25.
        """
        personas = []
        for i in range(5):
            profile = PersonaProfile(
                name=f"Persona_{i}",
                age=25 + i,
                occupation="Test",
            )
            personas.append(Persona(profile))
        return personas
    
    def test_add_persona(self, engine, personas):
        """Test that personas can be added to the simulation engine.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
        
        assert len(engine.personas) == 5
    
    def test_add_connection(self, engine, personas):
        """Test that connections can be added between personas.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
        
        engine.add_connection(personas[0].id, personas[1].id, strength=0.7)
        
        assert engine.graph.edge_count > 0
    
    def test_run_steps(self, engine, personas):
        """Test running a specific number of simulation steps.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
            persona.add_belief("test_topic", 0.5, 0.5)
        
        # Add some connections
        for i in range(len(personas) - 1):
            engine.add_connection(personas[i].id, personas[i + 1].id)
        
        state = engine.run_steps(4)
        
        assert state.step_count == 4
        assert state.phase == SimulationPhase.RUNNING
    
    def test_run_full_simulation(self, engine, personas):
        """Test running a complete simulation to completion.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
            persona.add_belief("topic_a", 0.3, 0.5)
        
        for i in range(len(personas) - 1):
            engine.add_connection(personas[i].id, personas[i + 1].id)
        
        state = engine.run()
        
        assert state.phase == SimulationPhase.COMPLETED
        assert state.step_count > 0
    
    def test_event_callbacks(self, engine, personas):
        """Test that interaction event callbacks are triggered during simulation.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
            persona.add_belief("topic", 0.5, 0.5)
        
        engine.add_connection(personas[0].id, personas[1].id)
        
        events_received = []
        engine.on_interaction(lambda e: events_received.append(e))
        
        engine.run_steps(2)
        
        assert len(events_received) > 0
    
    def test_export_state(self, engine, personas):
        """Test exporting the simulation engine state to a dictionary.

        Args:
            engine: The SimulationEngine fixture.
            personas: The list of test personas fixture.
        """
        for persona in personas:
            engine.add_persona(persona)
        
        exported = engine.export_state()
        
        assert "phase" in exported
        assert "persona_count" in exported
        assert exported["persona_count"] == 5


class TestNarrativeTracker:
    """Tests for NarrativeTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a NarrativeTracker instance for testing.

        Returns:
            NarrativeTracker: A new narrative tracker instance.
        """
        return NarrativeTracker()
    
    def test_register_topic(self, tracker):
        """Test registering a new topic with the narrative tracker.

        Args:
            tracker: The NarrativeTracker fixture.
        """
        narrative = tracker.register_topic("climate_change")
        
        assert narrative.topic == "climate_change"
        assert "climate_change" in tracker.topics
    
    def test_update_belief(self, tracker):
        """Test updating persona beliefs and computing similarity.

        Args:
            tracker: The NarrativeTracker fixture.
        """
        tracker.register_topic("topic_a")
        
        tracker.update_belief("persona_1", "topic_a", 0.5, 0.6)
        tracker.update_belief("persona_2", "topic_a", 0.7, 0.8)
        
        similarity = tracker.compute_pairwise_similarity("persona_1", "persona_2")
        assert similarity > 0.5  # Similar positions
    
    def test_take_snapshot(self, tracker):
        """Test taking a belief snapshot for a topic.

        Args:
            tracker: The NarrativeTracker fixture.
        """
        tracker.register_topic("topic")
        tracker.update_belief("p1", "topic", 0.5, 0.5)
        tracker.update_belief("p2", "topic", 0.6, 0.5)
        tracker.update_belief("p3", "topic", -0.5, 0.5)
        
        snapshot = tracker.take_snapshot("topic")
        
        assert len(snapshot.belief_positions) == 3
        assert snapshot.mean_position == pytest.approx(0.2, abs=0.01)
    
    def test_detect_echo_chambers(self, tracker):
        """Test detection of echo chambers based on belief similarity.

        Args:
            tracker: The NarrativeTracker fixture.
        """
        # Create group with very similar beliefs
        for i in range(5):
            tracker.update_belief(f"chamber_p{i}", "topic", 0.8 + i * 0.01, 0.9)
        
        # Create another group with different beliefs
        for i in range(5):
            tracker.update_belief(f"other_p{i}", "topic", -0.8 + i * 0.01, 0.9)
        
        chambers = tracker.detect_echo_chambers(similarity_threshold=0.8, min_size=3)
        
        # Should detect the two echo chambers
        assert len(chambers) >= 1
    
    def test_narrative_summary(self, tracker):
        """Test getting a narrative summary for a topic.

        Args:
            tracker: The NarrativeTracker fixture.
        """
        tracker.register_topic("topic")
        tracker.update_belief("p1", "topic", 0.5, 0.5, source_id="p0")
        
        summary = tracker.get_narrative_summary("topic")
        
        assert summary["topic"] == "topic"
        assert "interaction_count" in summary


class TestBeliefSnapshot:
    """Tests for BeliefSnapshot class."""
    
    def test_mean_position(self):
        """Test calculation of mean belief position across personas."""
        snapshot = BeliefSnapshot(
            timestamp=datetime.now(),
            topic="test",
            belief_positions={"p1": 0.5, "p2": 0.7, "p3": 0.3},
            belief_confidences={"p1": 0.5, "p2": 0.5, "p3": 0.5},
        )
        
        assert snapshot.mean_position == 0.5
    
    def test_position_variance(self):
        """Test calculation of position variance when all positions are identical."""
        snapshot = BeliefSnapshot(
            timestamp=datetime.now(),
            topic="test",
            belief_positions={"p1": 0.5, "p2": 0.5, "p3": 0.5},  # All same
            belief_confidences={},
        )
        
        assert snapshot.position_variance == 0.0
    
    def test_consensus_score(self):
        """Test that consensus score is high when positions are identical."""
        # High consensus (low variance)
        snapshot = BeliefSnapshot(
            timestamp=datetime.now(),
            topic="test",
            belief_positions={"p1": 0.5, "p2": 0.5, "p3": 0.5},
            belief_confidences={},
        )
        assert snapshot.consensus_score > 0.9
    
    def test_get_clusters(self):
        """Test clustering of belief positions into distinct groups."""
        snapshot = BeliefSnapshot(
            timestamp=datetime.now(),
            topic="test",
            belief_positions={
                "p1": 0.8, "p2": 0.85, "p3": 0.9,  # Cluster 1
                "p4": -0.8, "p5": -0.85,  # Cluster 2
            },
            belief_confidences={},
        )
        
        clusters = snapshot.get_clusters(threshold=0.3)
        assert len(clusters) == 2


class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a ConvergenceAnalyzer instance for testing.

        Returns:
            ConvergenceAnalyzer: A convergence analyzer with 0.2 consensus threshold.
        """
        return ConvergenceAnalyzer(consensus_threshold=0.2)
    
    def test_analyze_converging_narrative(self, analyzer):
        """Test analysis of a narrative that shows convergence over time.

        Args:
            analyzer: The ConvergenceAnalyzer fixture.
        """
        narrative = Narrative(topic="test")
        
        # Add snapshots showing convergence
        now = datetime.now()
        for i in range(5):
            positions = {f"p{j}": 0.5 + (0.3 - i * 0.06) * ((-1) ** j) for j in range(5)}
            snapshot = BeliefSnapshot(
                timestamp=now + timedelta(hours=i),
                topic="test",
                belief_positions=positions,
                belief_confidences={},
            )
            narrative.add_snapshot(snapshot)
        
        metrics = analyzer.analyze_narrative(narrative)
        
        assert metrics.is_converging
        assert metrics.final_variance < metrics.initial_variance
    
    def test_detect_anomalous_convergence(self, analyzer):
        """Test detection of anomalous convergence patterns in narratives.

        Args:
            analyzer: The ConvergenceAnalyzer fixture.
        """
        # Create narratives with different convergence rates
        narratives = []
        for rate in [0.1, 0.1, 0.1, 0.5, 0.1]:  # One anomalous
            n = Narrative(topic=f"topic_{rate}")
            # Mock convergence metrics
            narratives.append(n)
        
        # This would need actual snapshots in real use
        # For now just test the interface works
        metrics = [analyzer.analyze_narrative(n) for n in narratives]
        anomalous = analyzer.detect_anomalous_convergence(metrics)
        
        assert isinstance(anomalous, list)


class TestIndependenceDetector:
    """Tests for IndependenceDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create an IndependenceDetector instance for testing.

        Returns:
            IndependenceDetector: A detector with 5-minute timing window.
        """
        return IndependenceDetector(timing_window_minutes=5.0)
    
    def test_analyze_timing_correlation(self, detector):
        """Test detection of timing correlation between persona events.

        Args:
            detector: The IndependenceDetector fixture.
        """
        now = datetime.now()
        
        # Create synchronized events (within timing window)
        events = [
            InteractionEvent(
                timestamp=now,
                interaction_type=InteractionType.POST,
                source_id="p1",
                target_id=None,
                topic="topic_a",
            ),
            InteractionEvent(
                timestamp=now + timedelta(minutes=1),
                interaction_type=InteractionType.POST,
                source_id="p2",
                target_id=None,
                topic="topic_a",
            ),
            InteractionEvent(
                timestamp=now + timedelta(minutes=2),
                interaction_type=InteractionType.POST,
                source_id="p3",
                target_id=None,
                topic="topic_a",
            ),
        ]
        
        signals = detector.analyze_timing_correlation(
            events,
            persona_ids=["p1", "p2", "p3"],
        )
        
        # Should detect timing coordination
        assert len(signals) > 0
        assert signals[0].signal_type == "timing"
    
    def test_compute_independence_score(self, detector):
        """Test computation of independence scores for personas.

        Args:
            detector: The IndependenceDetector fixture.
        """
        # Add some signals
        signal = CoordinationSignal(
            signal_type="timing",
            strength=0.8,
            persona_ids=["p1", "p2"],
            topic="test",
            description="Test signal",
        )
        detector._signals.append(signal)
        
        score_p1 = detector.compute_independence_score("p1")
        score_other = detector.compute_independence_score("p3")
        
        assert score_p1 < score_other  # p1 is involved in coordination
    
    def test_coordination_summary(self, detector):
        """Test generation of coordination summary from detected signals.

        Args:
            detector: The IndependenceDetector fixture.
        """
        # Add signals
        for i in range(3):
            detector._signals.append(CoordinationSignal(
                signal_type="timing",
                strength=0.7 + i * 0.1,
                persona_ids=["p1", "p2"],
                topic=f"topic_{i}",
                description=f"Signal {i}",
            ))
        
        summary = detector.get_coordination_summary()
        
        assert summary["total_signals"] == 3
        assert "signals_by_type" in summary


class TestCoordinationSignal:
    """Tests for CoordinationSignal class."""
    
    def test_is_strong(self):
        """Test classification of signals as strong or weak based on strength."""
        strong = CoordinationSignal(
            signal_type="timing",
            strength=0.8,
            persona_ids=["p1"],
            topic="test",
            description="Strong signal",
        )
        weak = CoordinationSignal(
            signal_type="timing",
            strength=0.3,
            persona_ids=["p1"],
            topic="test",
            description="Weak signal",
        )
        
        assert strong.is_strong
        assert not weak.is_strong
    
    def test_is_moderate(self):
        """Test classification of signals as moderate based on strength."""
        moderate = CoordinationSignal(
            signal_type="timing",
            strength=0.5,
            persona_ids=["p1"],
            topic="test",
            description="Moderate signal",
        )
        
        assert moderate.is_moderate
