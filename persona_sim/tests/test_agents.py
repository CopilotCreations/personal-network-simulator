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
        profile = PersonaProfile(
            name="Alice",
            age=28,
            occupation="Developer",
            susceptibility=0.5,
        )
        return Persona(profile)
    
    def test_persona_has_unique_id(self, persona):
        profile = PersonaProfile(name="Bob", age=30, occupation="Designer")
        other_persona = Persona(profile)
        assert persona.id != other_persona.id
    
    def test_add_and_get_belief(self, persona):
        persona.add_belief("climate_change", 0.8, confidence=0.7)
        belief = persona.get_belief("climate_change")
        
        assert belief is not None
        assert belief.topic == "climate_change"
        assert belief.position == 0.8
        assert belief.confidence == 0.7
    
    def test_update_belief_with_reinforcement(self, persona):
        persona.add_belief("topic_a", 0.5, confidence=0.5)
        persona.update_belief("topic_a", 0.7, source_credibility=0.8)
        
        belief = persona.get_belief("topic_a")
        assert belief.position > 0.5  # Should move toward 0.7
        assert belief.confidence > 0.5  # Should increase with reinforcement
    
    def test_update_belief_with_contradiction(self, persona):
        persona.add_belief("topic_a", 0.5, confidence=0.5)
        persona.update_belief("topic_a", -0.5, source_credibility=0.8)
        
        belief = persona.get_belief("topic_a")
        # Position should move but confidence should decrease
        assert belief.confidence < 0.5
    
    def test_generate_response_templates(self, persona):
        persona.add_belief("test_topic", 0.8, confidence=0.7)
        response = persona.generate_response("test_topic")
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Positive belief should generate supportive response
        assert any(word in response.lower() for word in ["support", "important", "embrace"])
    
    def test_record_interaction(self, persona):
        assert persona.interaction_count == 0
        persona.record_interaction()
        assert persona.interaction_count == 1


class TestBelief:
    """Tests for Belief dataclass."""
    
    def test_belief_clamps_position(self):
        belief = Belief("topic", position=2.0, confidence=0.5)
        assert belief.position == 1.0
        
        belief2 = Belief("topic", position=-2.0, confidence=0.5)
        assert belief2.position == -1.0
    
    def test_belief_clamps_confidence(self):
        belief = Belief("topic", position=0.0, confidence=1.5)
        assert belief.confidence == 1.0


class TestMemory:
    """Tests for Memory class."""
    
    @pytest.fixture
    def memory(self):
        return Memory(max_detailed_memories=10, max_summaries=5)
    
    def test_add_memory(self, memory):
        memory.add_memory(
            interaction_type="received",
            source_id="persona_1",
            topic="test_topic",
            content_summary="Test content",
            sentiment=0.5,
        )
        assert memory.total_memories == 1
    
    def test_memory_is_bounded(self, memory):
        for i in range(20):
            memory.add_memory(
                interaction_type="received",
                source_id=f"persona_{i}",
                topic="topic",
                content_summary=f"Content {i}",
            )
        
        assert memory.total_memories <= memory.max_detailed_memories
    
    def test_get_recent_memories(self, memory):
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
        constraints = StyleConstraints(
            tone_level=ToneLevel.FORMAL,
            emotional_tone=EmotionalTone.ANALYTICAL,
            vocabulary_complexity=0.8,
            use_contractions=False,
        )
        return LinguisticStyle(constraints)
    
    @pytest.fixture
    def informal_style(self):
        constraints = StyleConstraints(
            tone_level=ToneLevel.INFORMAL,
            emotional_tone=EmotionalTone.PASSIONATE,
            vocabulary_complexity=0.3,
            use_contractions=True,
            use_emoji=True,
        )
        return LinguisticStyle(constraints)
    
    def test_apply_formal_style(self, formal_style):
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
        informal_style.set_seed(42)
        text = "I think this is good."
        styled = informal_style.apply_style(text)
        
        # Should have emoji
        assert any(char in styled for char in "👍✨💡🎯😕🤔⚠️📌💭📝")
    
    def test_style_similarity(self, formal_style, informal_style):
        similarity = LinguisticStyle.compute_similarity(formal_style, informal_style)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity < 0.8  # Should be somewhat different
    
    def test_same_style_similarity(self, formal_style):
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
        vector = formal_style.compute_style_vector()
        
        assert isinstance(vector, list)
        assert len(vector) == 7
        assert all(0.0 <= v <= 1.0 for v in vector)


class TestMemorySummary:
    """Tests for MemorySummary class."""
    
    def test_create_from_memories(self):
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
        summary = MemorySummary.from_memories("empty", [])
        
        assert summary.interaction_count == 0
        assert summary.average_sentiment == 0.0
