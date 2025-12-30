"""
Persona traits and constraints for synthetic agents.

Personas are stable profiles that define agent behavior without
accessing external APIs or platforms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import uuid


class PersonalityTrait(Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class PoliticalLeaning(Enum):
    """Simplified political spectrum for simulation."""
    FAR_LEFT = -2
    LEFT = -1
    CENTER = 0
    RIGHT = 1
    FAR_RIGHT = 2


@dataclass
class PersonaProfile:
    """
    Defines the stable characteristics of a synthetic persona.
    
    These traits remain constant throughout a simulation and
    influence how the persona interacts and responds.
    """
    name: str
    age: int
    occupation: str
    interests: List[str] = field(default_factory=list)
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    political_leaning: PoliticalLeaning = PoliticalLeaning.CENTER
    credibility_weight: float = 0.5  # How much others trust this persona
    susceptibility: float = 0.5  # How easily influenced by others
    
    def __post_init__(self):
        # Initialize default personality traits if not provided
        if not self.personality_traits:
            self.personality_traits = {
                trait: 0.5 for trait in PersonalityTrait
            }
        # Clamp values to valid ranges
        self.credibility_weight = max(0.0, min(1.0, self.credibility_weight))
        self.susceptibility = max(0.0, min(1.0, self.susceptibility))


@dataclass
class Belief:
    """A belief held by a persona on a specific topic."""
    topic: str
    position: float  # -1.0 to 1.0 (against to for)
    confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        self.position = max(-1.0, min(1.0, self.position))
        self.confidence = max(0.0, min(1.0, self.confidence))


class Persona:
    """
    A synthetic agent with stable profile, beliefs, and behavior.
    
    Personas interact with other personas through the simulation
    engine and can update their beliefs based on interactions.
    """
    
    def __init__(self, profile: PersonaProfile):
        self.id = str(uuid.uuid4())
        self.profile = profile
        self.beliefs: Dict[str, Belief] = {}
        self._interaction_count = 0
    
    @property
    def name(self) -> str:
        return self.profile.name
    
    def add_belief(self, topic: str, position: float, confidence: float = 0.5) -> None:
        """Add or update a belief on a topic."""
        self.beliefs[topic] = Belief(topic, position, confidence)
    
    def get_belief(self, topic: str) -> Optional[Belief]:
        """Get the persona's belief on a topic."""
        return self.beliefs.get(topic)
    
    def update_belief(self, topic: str, influence: float, source_credibility: float) -> None:
        """
        Update a belief based on external influence.
        
        The update is modulated by:
        - The persona's susceptibility
        - The source's credibility
        - The persona's current confidence in the belief
        """
        if topic not in self.beliefs:
            # Create new belief with low confidence
            self.beliefs[topic] = Belief(topic, influence, 0.3)
            return
        
        belief = self.beliefs[topic]
        
        # Calculate influence strength
        influence_strength = (
            self.profile.susceptibility * 
            source_credibility * 
            (1.0 - belief.confidence * 0.5)  # Higher confidence = less change
        )
        
        # Update position with bounded change
        position_delta = (influence - belief.position) * influence_strength * 0.1
        new_position = belief.position + position_delta
        
        # Confidence increases with reinforcement, decreases with contradiction
        if (influence > 0 and belief.position > 0) or (influence < 0 and belief.position < 0):
            new_confidence = min(1.0, belief.confidence + 0.05)
        else:
            new_confidence = max(0.1, belief.confidence - 0.02)
        
        self.beliefs[topic] = Belief(topic, new_position, new_confidence)
    
    def generate_response(self, topic: str, context: Optional[str] = None) -> str:
        """
        Generate a rule-based response on a topic.
        
        NOTE: This uses templates, not LLM generation, to ensure
        reproducibility and avoid external API dependencies.
        """
        belief = self.get_belief(topic)
        
        if belief is None:
            return self._neutral_response(topic)
        
        if belief.position > 0.5:
            return self._positive_response(topic, belief)
        elif belief.position < -0.5:
            return self._negative_response(topic, belief)
        else:
            return self._neutral_response(topic)
    
    def _positive_response(self, topic: str, belief: Belief) -> str:
        """Generate a supportive response template."""
        templates = [
            f"I strongly support {topic}.",
            f"{topic} is important and beneficial.",
            f"We should embrace {topic}.",
        ]
        idx = hash(self.id + topic) % len(templates)
        return templates[idx]
    
    def _negative_response(self, topic: str, belief: Belief) -> str:
        """Generate an opposing response template."""
        templates = [
            f"I have concerns about {topic}.",
            f"{topic} has significant drawbacks.",
            f"We should be cautious about {topic}.",
        ]
        idx = hash(self.id + topic) % len(templates)
        return templates[idx]
    
    def _neutral_response(self, topic: str) -> str:
        """Generate a neutral response template."""
        templates = [
            f"I'm still forming my opinion on {topic}.",
            f"{topic} has both pros and cons.",
            f"I'd like to learn more about {topic}.",
        ]
        idx = hash(self.id + topic) % len(templates)
        return templates[idx]
    
    def record_interaction(self) -> None:
        """Record that an interaction occurred."""
        self._interaction_count += 1
    
    @property
    def interaction_count(self) -> int:
        return self._interaction_count
    
    def __repr__(self) -> str:
        return f"Persona(name={self.name}, id={self.id[:8]})"
