"""Tests for network modules."""

import pytest
from datetime import datetime, timedelta

from persona_sim.network.graph import SocialGraph, Connection, ConnectionType
from persona_sim.network.dynamics import (
    InteractionScheduler,
    InteractionEvent,
    InteractionType,
    SchedulingConfig,
)


class TestSocialGraph:
    """Tests for SocialGraph class."""
    
    @pytest.fixture
    def graph(self):
        """Create a fresh SocialGraph instance for testing.

        Returns:
            SocialGraph: An empty social graph instance.
        """
        return SocialGraph()
    
    def test_add_nodes(self, graph):
        """Test that nodes can be added to the graph.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_node("persona_1")
        graph.add_node("persona_2")
        
        assert graph.node_count == 2
        assert "persona_1" in graph.nodes
    
    def test_add_connection(self, graph):
        """Test that connections can be added between nodes.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.FOLLOWER, strength=0.7)
        
        assert graph.edge_count == 1
        connection = graph.get_connection("a", "b")
        assert connection is not None
        assert connection.strength == 0.7
    
    def test_mutual_connection(self, graph):
        """Test that mutual connections create edges in both directions.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        
        assert graph.edge_count == 2
        assert graph.get_connection("a", "b") is not None
        assert graph.get_connection("b", "a") is not None
    
    def test_get_followers(self, graph):
        """Test retrieving followers of a node.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.FOLLOWER)
        graph.add_connection("c", "b", ConnectionType.FOLLOWER)
        
        followers = graph.get_followers("b")
        assert set(followers) == {"a", "c"}
    
    def test_get_following(self, graph):
        """Test retrieving nodes that a given node follows.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.FOLLOWER)
        graph.add_connection("a", "c", ConnectionType.FOLLOWER)
        
        following = graph.get_following("a")
        assert set(following) == {"b", "c"}
    
    def test_get_mutual_connections(self, graph):
        """Test retrieving mutual connections excludes one-way connections.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        graph.add_connection("a", "c", ConnectionType.FOLLOWER)  # One-way
        
        mutual = graph.get_mutual_connections("a")
        assert "b" in mutual
        assert "c" not in mutual
    
    def test_remove_node(self, graph):
        """Test that removing a node also removes its connections.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        graph.add_connection("a", "c", ConnectionType.MUTUAL)
        
        graph.remove_node("a")
        
        assert "a" not in graph.nodes
        assert graph.get_connection("a", "b") is None
        assert graph.get_connection("b", "a") is None
    
    def test_graph_density(self, graph):
        """Test that a fully connected graph has density 1.0.

        Args:
            graph: The SocialGraph fixture instance.
        """
        # Complete graph with 3 nodes has 6 directed edges
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        graph.add_connection("b", "c", ConnectionType.MUTUAL)
        graph.add_connection("a", "c", ConnectionType.MUTUAL)
        
        density = graph.density()
        assert density == 1.0  # Fully connected
    
    def test_clustering_coefficient(self, graph):
        """Test clustering coefficient is 1.0 when all neighbors are connected.

        Args:
            graph: The SocialGraph fixture instance.
        """
        # Triangle: all neighbors connected
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        graph.add_connection("b", "c", ConnectionType.MUTUAL)
        graph.add_connection("a", "c", ConnectionType.MUTUAL)
        
        cc = graph.clustering_coefficient("a")
        assert cc == 1.0  # All neighbors connected
    
    def test_shortest_path(self, graph):
        """Test shortest path length calculation between nodes.

        Args:
            graph: The SocialGraph fixture instance.
        """
        graph.add_connection("a", "b", ConnectionType.FOLLOWER)
        graph.add_connection("b", "c", ConnectionType.FOLLOWER)
        graph.add_connection("c", "d", ConnectionType.FOLLOWER)
        
        assert graph.shortest_path_length("a", "d") == 3
        assert graph.shortest_path_length("a", "a") == 0
        assert graph.shortest_path_length("d", "a") is None  # No reverse path
    
    def test_find_communities(self, graph):
        """Test that disconnected groups are identified as separate communities.

        Args:
            graph: The SocialGraph fixture instance.
        """
        # Create two disconnected groups
        graph.add_connection("a", "b", ConnectionType.MUTUAL)
        graph.add_connection("b", "c", ConnectionType.MUTUAL)
        
        graph.add_connection("x", "y", ConnectionType.MUTUAL)
        graph.add_connection("y", "z", ConnectionType.MUTUAL)
        
        communities = graph.find_communities(min_size=2)
        assert len(communities) == 2
    
    def test_create_random_graph(self):
        """Test creating a random graph with specified connection probability."""
        personas = [f"p{i}" for i in range(10)]
        graph = SocialGraph.create_random(personas, connection_probability=0.5, seed=42)
        
        assert graph.node_count == 10
        assert graph.edge_count > 0
    
    def test_create_small_world_graph(self):
        """Test creating a small-world graph with proper connectivity."""
        personas = [f"p{i}" for i in range(10)]
        graph = SocialGraph.create_small_world(personas, k=4, rewire_prob=0.1, seed=42)
        
        assert graph.node_count == 10
        # Each node should have approximately k connections
        for persona in personas:
            neighbors = len(graph.get_neighbors(persona))
            assert neighbors >= 2  # At least some connections


class TestConnection:
    """Tests for Connection class."""
    
    def test_record_interaction(self):
        """Test that recording an interaction increases connection strength."""
        conn = Connection("a", "b", strength=0.5)
        initial_strength = conn.strength
        
        conn.record_interaction()
        
        assert conn.interaction_count == 1
        assert conn.strength > initial_strength
    
    def test_decay(self):
        """Test that decay reduces connection strength by the given factor."""
        conn = Connection("a", "b", strength=0.5)
        conn.decay(factor=0.9)
        
        assert conn.strength == pytest.approx(0.45)
    
    def test_strength_clamping(self):
        """Test that connection strength is clamped between 0.0 and 1.0."""
        conn = Connection("a", "b", strength=1.5)
        assert conn.strength == 1.0
        
        conn2 = Connection("a", "b", strength=-0.5)
        assert conn2.strength == 0.0


class TestInteractionScheduler:
    """Tests for InteractionScheduler class."""
    
    @pytest.fixture
    def scheduler(self):
        """Create an InteractionScheduler instance for testing.

        Returns:
            InteractionScheduler: A scheduler configured with test settings.
        """
        config = SchedulingConfig(
            base_interval_minutes=30.0,
            interval_variance=0.5,
        )
        return InteractionScheduler(config, seed=42)
    
    def test_schedule_event(self, scheduler):
        """Test scheduling a single interaction event.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        event = scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="persona_1",
            topic="test_topic",
            delay_minutes=10,
        )
        
        assert event.source_id == "persona_1"
        assert event.topic == "test_topic"
        assert scheduler.pending_count == 1
    
    def test_process_next_event(self, scheduler):
        """Test that events are processed in chronological order.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="persona_1",
            topic="topic_1",
            delay_minutes=5,
        )
        scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="persona_2",
            topic="topic_2",
            delay_minutes=10,
        )
        
        # Should get earliest event first
        event = scheduler.process_next_event()
        assert event.source_id == "persona_1"
        assert scheduler.completed_count == 1
        assert scheduler.pending_count == 1
    
    def test_schedule_burst(self, scheduler):
        """Test scheduling a burst of events from a single source.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        events = scheduler.schedule_burst(
            source_id="persona_1",
            topic="hot_topic",
        )
        
        assert len(events) >= 3  # Minimum burst size
        # All events should be from same source
        assert all(e.source_id == "persona_1" for e in events)
    
    def test_schedule_coordinated_response(self, scheduler):
        """Test scheduling coordinated responses that reference an original event.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        original = scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="leader",
            topic="topic",
            delay_minutes=0,
        )
        
        responses = scheduler.schedule_coordinated_response(
            responder_ids=["follower_1", "follower_2", "follower_3"],
            original_event=original,
            topic="topic",
        )
        
        assert len(responses) == 3
        # All responses should reference original event
        assert all(r.in_response_to == original.event_id for r in responses)
    
    def test_process_events_until(self, scheduler):
        """Test processing only events scheduled before a given time.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        start = scheduler.current_time
        
        scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="p1",
            topic="t1",
            delay_minutes=10,
        )
        scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="p2",
            topic="t2",
            delay_minutes=30,
        )
        scheduler.schedule_event(
            interaction_type=InteractionType.POST,
            source_id="p3",
            topic="t3",
            delay_minutes=60,
        )
        
        end_time = start + timedelta(minutes=45)
        processed = scheduler.process_events_until(end_time)
        
        assert len(processed) == 2  # Only events before 45 minutes
        assert scheduler.pending_count == 1
    
    def test_timing_statistics(self, scheduler):
        """Test that timing statistics are calculated correctly.

        Args:
            scheduler: The InteractionScheduler fixture instance.
        """
        # Schedule multiple events with known timing
        for i in range(5):
            scheduler.schedule_event(
                interaction_type=InteractionType.POST,
                source_id="persona_1",
                topic="topic",
                delay_minutes=i * 10,
            )
        
        # Process all events
        end_time = scheduler.current_time + timedelta(hours=1)
        scheduler.process_events_until(end_time)
        
        stats = scheduler.get_timing_statistics("persona_1")
        
        assert stats["total_events"] == 5
        assert stats["avg_interval_minutes"] == pytest.approx(10.0)


class TestInteractionEvent:
    """Tests for InteractionEvent class."""
    
    def test_event_has_unique_id(self):
        """Test that each event receives a unique identifier."""
        event1 = InteractionEvent(
            timestamp=datetime.now(),
            interaction_type=InteractionType.POST,
            source_id="p1",
            target_id=None,
            topic="topic",
        )
        event2 = InteractionEvent(
            timestamp=datetime.now(),
            interaction_type=InteractionType.POST,
            source_id="p1",
            target_id=None,
            topic="topic",
        )
        
        assert event1.event_id != event2.event_id
    
    def test_event_ordering(self):
        """Test that events are ordered chronologically by timestamp."""
        now = datetime.now()
        earlier = InteractionEvent(
            timestamp=now,
            interaction_type=InteractionType.POST,
            source_id="p1",
            target_id=None,
            topic="topic",
        )
        later = InteractionEvent(
            timestamp=now + timedelta(minutes=10),
            interaction_type=InteractionType.POST,
            source_id="p2",
            target_id=None,
            topic="topic",
        )
        
        assert earlier < later
