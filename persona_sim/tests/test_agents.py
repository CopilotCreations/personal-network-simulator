"""Tests for persona and agent modules."""

import pytest
from datetime import datetime

from persona_sim.agents.persona import (
    Persona,
    PersonaProfile,
    PersonalityTrait,
    PoliticalLeaning,
    Belief,
)
from persona_sim.agents.memory import Memory, MemoryItem, MemorySummary
from persona_sim.agents.style import (
    LinguisticStyle,
    StyleConstraints,
    ToneLevel,
    EmotionalTone,
)


class TestPersonaProfile:
    """Tests for PersonaProfile dataclass."""
    
    def test_create_basic_profile(self):
        """Test creating a basic PersonaProfile with required fields only.

        Verifies that default values are correctly assigned for optional fields
        like credibility_weight and susceptibility.
        """
        profile = PersonaProfile(
            name="Test User",
            age=30,
            occupation="Engineer",
        )
        assert profile.name == "Test User"
        assert profile.age == 30
        assert profile.credibility_weight == 0.5
        assert profile.susceptibility == 0.5
    
    def test_profile_with_traits(self):
        """Test creating a PersonaProfile with all optional traits specified.

        Verifies that interests, political_leaning, credibility_weight, and
        susceptibility are correctly stored when provided.
        """
        profile = PersonaProfile(
            name="Test User",
            age=25,
            occupation="Artist",
            interests=["painting", "music"],
            political_leaning=PoliticalLeaning.LEFT,
            credibility_weight=0.8,
            susceptibility=0.3,
        )
        assert profile.political_leaning == PoliticalLeaning.LEFT
        assert len(profile.interests) == 2
        assert profile.credibility_weight == 0.8
    
    def test_profile_clamps_values(self):
        """Test that PersonaProfile clamps out-of-range values to valid bounds.

        Verifies that credibility_weight is clamped to [0.0, 1.0] and
        susceptibility is clamped to [0.0, 1.0].
        """
        profile = PersonaProfile(
            name="Test",
            age=30,
            occupation="Test",
            credibility_weight=1.5,  # Should be clamped to 1.0
            susceptibility=-0.5,  # Should be clamped to 0.0
        )
        assert profile.credibility_weight == 1.0
        assert profile.susceptibility == 0.0


class TestPersona:
    """Tests for Persona class."""
    
    @pytest.fixture
    def persona(self):
        """Create a test Persona fixture for use in persona tests.

        Returns:
            Persona: A Persona instance with a basic profile.
        """
        profile = PersonaProfile(
            name="Alice",
            age=28,
            occupation="Developer",
            susceptibility=0.5,
        )
        return Persona(profile)
    
    def test_persona_has_unique_id(self, persona):
        """Test that each Persona instance receives a unique identifier.

        Args:
            persona: The persona fixture to test against.
        """
        profile = PersonaProfile(name="Bob", age=30, occupation="Designer")
        other_persona = Persona(profile)
        assert persona.id != other_persona.id
    
    def test_add_and_get_belief(self, persona):
        """Test adding a belief to a persona and retrieving it.

        Args:
            persona: The persona fixture to test against.
        """
        persona.add_belief("climate_change", 0.8, confidence=0.7)
        belief = persona.get_belief("climate_change")
        
        assert belief is not None
        assert belief.topic == "climate_change"
        assert belief.position == 0.8
        assert belief.confidence == 0.7
    
    def test_update_belief_with_reinforcement(self, persona):
        """Test that beliefs move toward reinforcing information.

        Args:
            persona: The persona fixture to test against.

        When new information reinforces existing beliefs, both position and
        confidence should increase.
        """
        persona.add_belief("topic_a", 0.5, confidence=0.5)
        persona.update_belief("topic_a", 0.7, source_credibility=0.8)
        
        belief = persona.get_belief("topic_a")
        assert belief.position > 0.5  # Should move toward 0.7
        assert belief.confidence > 0.5  # Should increase with reinforcement
    
    def test_update_belief_with_contradiction(self, persona):
        """Test that contradictory information decreases belief confidence.

        Args:
            persona: The persona fixture to test against.

        When new information contradicts existing beliefs, confidence should
        decrease even if the position changes.
        """
        persona.add_belief("topic_a", 0.5, confidence=0.5)
        persona.update_belief("topic_a", -0.5, source_credibility=0.8)
        
        belief = persona.get_belief("topic_a")
        # Position should move but confidence should decrease
        assert belief.confidence < 0.5
    
    def test_generate_response_templates(self, persona):
        """Test that personas generate appropriate responses based on beliefs.

        Args:
            persona: The persona fixture to test against.

        A positive belief should produce a supportive response containing
        keywords like 'support', 'important', or 'embrace'.
        """
        persona.add_belief("test_topic", 0.8, confidence=0.7)
        response = persona.generate_response("test_topic")
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Positive belief should generate supportive response
        assert any(word in response.lower() for word in ["support", "important", "embrace"])
    
    def test_record_interaction(self, persona):
        """Test that recording an interaction increments the interaction count.

        Args:
            persona: The persona fixture to test against.
        """
        assert persona.interaction_count == 0
        persona.record_interaction()
        assert persona.interaction_count == 1


class TestBelief:
    """Tests for Belief dataclass."""
    
    def test_belief_clamps_position(self):
        """Test that Belief clamps position values to [-1.0, 1.0] range.

        Values above 1.0 should be clamped to 1.0, and values below -1.0
        should be clamped to -1.0.
        """
        belief = Belief("topic", position=2.0, confidence=0.5)
        assert belief.position == 1.0
        
        belief2 = Belief("topic", position=-2.0, confidence=0.5)
        assert belief2.position == -1.0
    
    def test_belief_clamps_confidence(self):
        """Test that Belief clamps confidence values to [0.0, 1.0] range.

        Values above 1.0 should be clamped to 1.0.
        """
        belief = Belief("topic", position=0.0, confidence=1.5)
        assert belief.confidence == 1.0


class TestMemory:
    """Tests for Memory class."""
    
    @pytest.fixture
    def memory(self):
        """Create a Memory fixture with limited capacity for testing.

        Returns:
            Memory: A Memory instance with max 10 detailed memories and 5 summaries.
        """
        return Memory(max_detailed_memories=10, max_summaries=5)
    
    def test_add_memory(self, memory):
        """Test adding a single memory item to the Memory store.

        Args:
            memory: The memory fixture to test against.
        """
        memory.add_memory(
            interaction_type="received",
            source_id="persona_1",
            topic="test_topic",
            content_summary="Test content",
            sentiment=0.5,
        )
        assert memory.total_memories == 1
    
    def test_memory_is_bounded(self, memory):
        """Test that Memory does not exceed its maximum capacity.

        Args:
            memory: The memory fixture to test against.

        When more memories are added than the maximum, old memories should be
        evicted to maintain the size limit.
        """
        for i in range(20):
            memory.add_memory(
                interaction_type="received",
                source_id=f"persona_{i}",
                topic="topic",
                content_summary=f"Content {i}",
            )
        
        assert memory.total_memories <= memory.max_detailed_memories
    
    def test_get_recent_memories(self, memory):
        """Test retrieving the most recent memories.

        Args:
            memory: The memory fixture to test against.

        Should return exactly the requested number of recent memories.
        """
        for i in range(5):
            memory.add_memory(
                interaction_type="sent",
                source_id="me",
                topic=f"topic_{i}",
                content_summary=f"Content {i}",
            )
        
        recent = memory.get_recent_memories(3)
        assert len(recent) == 3
    
    def test_topic_summarization(self, memory):
        """Test that memories on the same topic get summarized.

        Args:
            memory: The memory fixture to test against.

        Adding many memories on the same topic should trigger automatic
        summarization with correct interaction counts.
        """
        # Add many memories on same topic to trigger summarization
        for i in range(15):
            memory.add_memory(
                interaction_type="received",
                source_id=f"persona_{i}",
                topic="hot_topic",
                content_summary=f"Discussion {i}",
                sentiment=0.3,
            )
        
        summary = memory.get_topic_summary("hot_topic")
        assert summary is not None
        assert summary.interaction_count > 0
    
    def test_compute_fingerprint(self, memory):
        """Test computing an MD5 fingerprint of memory contents.

        Args:
            memory: The memory fixture to test against.

        The fingerprint should be a 32-character hexadecimal string (MD5 hash).
        """
        memory.add_memory(
            interaction_type="received",
            source_id="source",
            topic="topic",
            content_summary="content",
        )
        
        fingerprint = memory.compute_fingerprint()
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 32  # MD5 hex length


class TestLinguisticStyle:
    """Tests for LinguisticStyle class."""
    
    @pytest.fixture
    def formal_style(self):
        """Create a formal LinguisticStyle fixture for testing.

        Returns:
            LinguisticStyle: A formal, analytical style without contractions.
        """
        constraints = StyleConstraints(
            tone_level=ToneLevel.FORMAL,
            emotional_tone=EmotionalTone.ANALYTICAL,
            vocabulary_complexity=0.8,
            use_contractions=False,
        )
        return LinguisticStyle(constraints)
    
    @pytest.fixture
    def informal_style(self):
        """Create an informal LinguisticStyle fixture for testing.

        Returns:
            LinguisticStyle: An informal, passionate style with contractions and emoji.
        """
        constraints = StyleConstraints(
            tone_level=ToneLevel.INFORMAL,
            emotional_tone=EmotionalTone.PASSIONATE,
            vocabulary_complexity=0.3,
            use_contractions=True,
            use_emoji=True,
        )
        return LinguisticStyle(constraints)
    
    def test_apply_formal_style(self, formal_style):
        """Test applying formal style adds appropriate formal openers.

        Args:
            formal_style: The formal style fixture to test against.
        """
        formal_style.set_seed(42)
        text = "I think this is good."
        styled = formal_style.apply_style(text)
        
        # Should add formal opener
        assert any(phrase in styled for phrase in [
            "In my assessment",
            "It appears that",
            "One might argue",
            "Evidence suggests",
            "It is worth noting",
        ])
    
    def test_apply_informal_style(self, informal_style):
        """Test applying informal style adds emoji to text.

        Args:
            informal_style: The informal style fixture to test against.
        """
        informal_style.set_seed(42)
        text = "I think this is good."
        styled = informal_style.apply_style(text)
        
        # Should have emoji
        assert any(char in styled for char in "👍✨💡🎯😕🤔⚠️📌💭📝")
    
    def test_style_similarity(self, formal_style, informal_style):
        """Test computing similarity between different styles.

        Args:
            formal_style: The formal style fixture.
            informal_style: The informal style fixture.

        Formal and informal styles should have low similarity (< 0.8).
        """
        similarity = LinguisticStyle.compute_similarity(formal_style, informal_style)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.8  # Should be somewhat different
    
    def test_same_style_similarity(self, formal_style):
        """Test that identical styles have very high similarity.

        Args:
            formal_style: The formal style fixture to compare against.

        Two styles with identical constraints should have similarity > 0.95.
        """
        constraints = StyleConstraints(
            tone_level=ToneLevel.FORMAL,
            emotional_tone=EmotionalTone.ANALYTICAL,
            vocabulary_complexity=0.8,
            use_contractions=False,
        )
        same_style = LinguisticStyle(constraints)
        
        similarity = LinguisticStyle.compute_similarity(formal_style, same_style)
        assert similarity > 0.95  # Should be nearly identical
    
    def test_compute_style_vector(self, formal_style):
        """Test computing a numeric style vector from a LinguisticStyle.

        Args:
            formal_style: The formal style fixture to test against.

        The style vector should be a list of 7 values, each in [0.0, 1.0].
        """
        vector = formal_style.compute_style_vector()
        
        assert isinstance(vector, list)
        assert len(vector) == 7
        assert all(0.0 <= v <= 1.0 for v in vector)


class TestMemorySummary:
    """Tests for MemorySummary class."""
    
    def test_create_from_memories(self):
        """Test creating a MemorySummary from a list of MemoryItems.

        Verifies that the summary correctly aggregates interaction count and
        computes average sentiment from the input memories.
        """
        memories = [
            MemoryItem(
                timestamp=datetime.now(),
                interaction_type="received",
                source_id="persona_1",
                topic="test",
                content_summary="Content 1",
                sentiment=0.5,
                importance=0.5,
            ),
            MemoryItem(
                timestamp=datetime.now(),
                interaction_type="received",
                source_id="persona_2",
                topic="test",
                content_summary="Content 2",
                sentiment=0.7,
                importance=0.6,
            ),
        ]
        
        summary = MemorySummary.from_memories("test", memories)
        
        assert summary.topic == "test"
        assert summary.interaction_count == 2
        assert summary.average_sentiment == 0.6  # (0.5 + 0.7) / 2
    
    def test_empty_summary(self):
        """Test creating a MemorySummary from an empty list of memories.

        An empty summary should have zero interaction count and zero average
        sentiment.
        """
        summary = MemorySummary.from_memories("empty", [])
        
        assert summary.interaction_count == 0
        assert summary.average_sentiment == 0.0
