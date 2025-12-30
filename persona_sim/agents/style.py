"""
Linguistic fingerprints and style constraints for personas.

Each persona has a consistent linguistic style that can be used
to detect coordination through phrasing similarity analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum
import random
import re


class ToneLevel(Enum):
    """Tone levels for communication style."""
    VERY_FORMAL = 0
    FORMAL = 1
    NEUTRAL = 2
    INFORMAL = 3
    VERY_INFORMAL = 4


class EmotionalTone(Enum):
    """Emotional undertones in communication."""
    ANALYTICAL = "analytical"
    PASSIONATE = "passionate"
    CALM = "calm"
    AGGRESSIVE = "aggressive"
    EMPATHETIC = "empathetic"


@dataclass
class StyleConstraints:
    """
    Constraints that define a persona's linguistic style.
    
    These are used to generate consistent, distinguishable
    communication patterns for each persona.
    """
    tone_level: ToneLevel = ToneLevel.NEUTRAL
    emotional_tone: EmotionalTone = EmotionalTone.CALM
    avg_sentence_length: int = 15  # words
    vocabulary_complexity: float = 0.5  # 0.0 = simple, 1.0 = complex
    use_contractions: bool = True
    use_emoji: bool = False
    punctuation_style: str = "standard"  # "minimal", "standard", "expressive"
    favorite_phrases: List[str] = field(default_factory=list)
    avoided_words: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.avg_sentence_length = max(5, min(40, self.avg_sentence_length))
        self.vocabulary_complexity = max(0.0, min(1.0, self.vocabulary_complexity))


class LinguisticStyle:
    """
    Applies linguistic style constraints to text generation.
    
    This class transforms template-based responses to match
    a persona's unique communication style.
    """
    
    # Word substitutions by complexity level
    SIMPLE_TO_COMPLEX = {
        "good": ["beneficial", "advantageous", "favorable"],
        "bad": ["detrimental", "adverse", "unfavorable"],
        "big": ["substantial", "significant", "considerable"],
        "small": ["minimal", "negligible", "modest"],
        "think": ["believe", "consider", "maintain"],
        "help": ["assist", "facilitate", "support"],
        "show": ["demonstrate", "illustrate", "indicate"],
        "use": ["utilize", "employ", "leverage"],
    }
    
    FORMAL_PHRASES = [
        "In my assessment,",
        "It appears that",
        "One might argue that",
        "Evidence suggests that",
        "It is worth noting that",
    ]
    
    INFORMAL_PHRASES = [
        "I think",
        "Honestly,",
        "Look,",
        "Here's the thing:",
        "Basically,",
    ]
    
    def __init__(self, constraints: StyleConstraints):
        self.constraints = constraints
        self._rng = random.Random()
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible styling."""
        self._rng.seed(seed)
    
    def apply_style(self, text: str) -> str:
        """
        Apply linguistic style constraints to text.
        
        This transforms the input text to match the persona's
        characteristic communication style.
        """
        text = self._apply_tone_level(text)
        text = self._apply_vocabulary_complexity(text)
        text = self._apply_contractions(text)
        text = self._apply_punctuation_style(text)
        text = self._apply_favorite_phrases(text)
        
        if self.constraints.use_emoji:
            text = self._add_emoji(text)
        
        return text
    
    def _apply_tone_level(self, text: str) -> str:
        """Adjust formality level of text."""
        tone = self.constraints.tone_level
        
        if tone in (ToneLevel.VERY_FORMAL, ToneLevel.FORMAL):
            # Add formal opener if none exists
            if not any(text.startswith(p) for p in self.FORMAL_PHRASES):
                prefix = self._rng.choice(self.FORMAL_PHRASES)
                text = f"{prefix} {text[0].lower()}{text[1:]}"
        
        elif tone in (ToneLevel.INFORMAL, ToneLevel.VERY_INFORMAL):
            # Add informal opener
            if not any(text.startswith(p) for p in self.INFORMAL_PHRASES):
                prefix = self._rng.choice(self.INFORMAL_PHRASES)
                text = f"{prefix} {text[0].lower()}{text[1:]}"
        
        return text
    
    def _apply_vocabulary_complexity(self, text: str) -> str:
        """Adjust vocabulary complexity."""
        if self.constraints.vocabulary_complexity > 0.7:
            # Use more complex vocabulary
            for simple, complex_options in self.SIMPLE_TO_COMPLEX.items():
                pattern = rf"\b{simple}\b"
                if re.search(pattern, text, re.IGNORECASE):
                    replacement = self._rng.choice(complex_options)
                    text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        
        return text
    
    def _apply_contractions(self, text: str) -> str:
        """Apply or remove contractions based on style."""
        contractions = {
            "I am": "I'm",
            "I have": "I've",
            "I will": "I'll",
            "I would": "I'd",
            "do not": "don't",
            "does not": "doesn't",
            "is not": "isn't",
            "are not": "aren't",
            "cannot": "can't",
            "will not": "won't",
            "would not": "wouldn't",
            "should not": "shouldn't",
            "could not": "couldn't",
            "it is": "it's",
            "that is": "that's",
            "there is": "there's",
            "we are": "we're",
            "they are": "they're",
        }
        
        if self.constraints.use_contractions:
            for full, contracted in contractions.items():
                text = re.sub(rf"\b{full}\b", contracted, text, flags=re.IGNORECASE)
        else:
            # Expand contractions
            for full, contracted in contractions.items():
                text = re.sub(rf"\b{contracted}\b", full, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_punctuation_style(self, text: str) -> str:
        """Adjust punctuation style."""
        style = self.constraints.punctuation_style
        
        if style == "expressive":
            # Add emphasis
            if text.endswith("."):
                if self._rng.random() > 0.5:
                    text = text[:-1] + "!"
        elif style == "minimal":
            # Remove excessive punctuation
            text = re.sub(r"!+", ".", text)
            text = re.sub(r"\?+", "?", text)
        
        return text
    
    def _apply_favorite_phrases(self, text: str) -> str:
        """Occasionally insert favorite phrases."""
        if self.constraints.favorite_phrases and self._rng.random() > 0.7:
            phrase = self._rng.choice(self.constraints.favorite_phrases)
            text = f"{phrase} {text}"
        
        return text
    
    def _add_emoji(self, text: str) -> str:
        """Add appropriate emoji to text."""
        positive_emoji = ["👍", "✨", "💡", "🎯"]
        negative_emoji = ["😕", "🤔", "⚠️"]
        neutral_emoji = ["📌", "💭", "📝"]
        
        # Simple sentiment detection
        positive_words = ["support", "agree", "good", "great", "beneficial"]
        negative_words = ["concern", "problem", "issue", "bad", "harmful"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in positive_words):
            emoji = self._rng.choice(positive_emoji)
        elif any(word in text_lower for word in negative_words):
            emoji = self._rng.choice(negative_emoji)
        else:
            emoji = self._rng.choice(neutral_emoji)
        
        return f"{text} {emoji}"
    
    def compute_style_vector(self) -> List[float]:
        """
        Compute a numerical representation of this style.
        
        Used for comparing style similarity between personas.
        """
        return [
            self.constraints.tone_level.value / 4.0,
            hash(self.constraints.emotional_tone.value) % 100 / 100.0,
            self.constraints.avg_sentence_length / 40.0,
            self.constraints.vocabulary_complexity,
            1.0 if self.constraints.use_contractions else 0.0,
            1.0 if self.constraints.use_emoji else 0.0,
            {"minimal": 0.0, "standard": 0.5, "expressive": 1.0}.get(
                self.constraints.punctuation_style, 0.5
            ),
        ]
    
    @staticmethod
    def compute_similarity(style1: "LinguisticStyle", style2: "LinguisticStyle") -> float:
        """
        Compute similarity between two linguistic styles.
        
        Returns a value from 0.0 (completely different) to 1.0 (identical).
        """
        vec1 = style1.compute_style_vector()
        vec2 = style2.compute_style_vector()
        
        # Euclidean distance normalized to similarity
        squared_diff = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
        max_distance = len(vec1)  # Maximum possible distance
        similarity = 1.0 - (squared_diff ** 0.5) / (max_distance ** 0.5)
        
        return max(0.0, min(1.0, similarity))
    
    def __repr__(self) -> str:
        return f"LinguisticStyle(tone={self.constraints.tone_level.name}, complexity={self.constraints.vocabulary_complexity:.2f})"
