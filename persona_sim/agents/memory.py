"""
Rolling memory summaries for synthetic agents.

Memory is intentionally lossy and bounded to simulate realistic
information retention and prevent unbounded state growth.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque
import hashlib


@dataclass
class MemoryItem:
    """A single memory item from an interaction."""
    timestamp: datetime
    interaction_type: str  # "received", "sent", "observed"
    source_id: str
    topic: str
    content_summary: str
    sentiment: float  # -1.0 to 1.0
    importance: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory item to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all memory item fields with
                timestamp converted to ISO format string.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "interaction_type": self.interaction_type,
            "source_id": self.source_id,
            "topic": self.topic,
            "content_summary": self.content_summary,
            "sentiment": self.sentiment,
            "importance": self.importance,
        }


@dataclass
class MemorySummary:
    """
    A compressed summary of multiple memory items.
    
    Used to maintain bounded memory while preserving key information.
    """
    topic: str
    interaction_count: int
    average_sentiment: float
    key_sources: List[str]
    last_updated: datetime
    summary_text: str
    
    @classmethod
    def from_memories(cls, topic: str, memories: List[MemoryItem]) -> "MemorySummary":
        """Create a summary from a list of memory items.

        Args:
            topic: The topic string to summarize memories for.
            memories: List of MemoryItem objects to summarize.

        Returns:
            MemorySummary: A new summary containing aggregated statistics
                and a generated summary text.
        """
        if not memories:
            return cls(
                topic=topic,
                interaction_count=0,
                average_sentiment=0.0,
                key_sources=[],
                last_updated=datetime.now(),
                summary_text=f"No memories about {topic}.",
            )
        
        avg_sentiment = sum(m.sentiment for m in memories) / len(memories)
        sources = list(set(m.source_id for m in memories))[:5]  # Top 5 sources
        latest = max(m.timestamp for m in memories)
        
        # Generate simple summary text
        sentiment_word = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "mixed"
        summary = f"Had {len(memories)} interactions about {topic} with {sentiment_word} sentiment."
        
        return cls(
            topic=topic,
            interaction_count=len(memories),
            average_sentiment=avg_sentiment,
            key_sources=sources,
            last_updated=latest,
            summary_text=summary,
        )


class Memory:
    """
    Bounded, lossy memory system for a persona.
    
    Maintains:
    - Recent detailed memories (bounded queue)
    - Compressed summaries by topic
    - Simple embedding-like fingerprints for similarity
    """
    
    def __init__(self, max_detailed_memories: int = 50, max_summaries: int = 20):
        """Initialize a new Memory instance.

        Args:
            max_detailed_memories: Maximum number of detailed memories to retain.
                Defaults to 50.
            max_summaries: Maximum number of topic summaries to retain.
                Defaults to 20.
        """
        self.max_detailed_memories = max_detailed_memories
        self.max_summaries = max_summaries
        self._detailed_memories: deque = deque(maxlen=max_detailed_memories)
        self._summaries: Dict[str, MemorySummary] = {}
        self._topic_memories: Dict[str, List[MemoryItem]] = {}
    
    def add_memory(
        self,
        interaction_type: str,
        source_id: str,
        topic: str,
        content_summary: str,
        sentiment: float = 0.0,
        importance: float = 0.5,
    ) -> None:
        """Add a new memory item.

        Args:
            interaction_type: Type of interaction ("received", "sent", "observed").
            source_id: Identifier of the interaction source.
            topic: Topic of the memory.
            content_summary: Brief summary of the content.
            sentiment: Sentiment score from -1.0 to 1.0. Defaults to 0.0.
            importance: Importance score from 0.0 to 1.0. Defaults to 0.5.
        """
        memory = MemoryItem(
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            source_id=source_id,
            topic=topic,
            content_summary=content_summary,
            sentiment=sentiment,
            importance=importance,
        )
        
        self._detailed_memories.append(memory)
        
        # Track by topic for summarization
        if topic not in self._topic_memories:
            self._topic_memories[topic] = []
        self._topic_memories[topic].append(memory)
        
        # Trigger summarization if topic has too many memories
        if len(self._topic_memories[topic]) > 10:
            self._summarize_topic(topic)
    
    def _summarize_topic(self, topic: str) -> None:
        """Compress memories for a topic into a summary.

        Args:
            topic: The topic string to summarize.
        """
        memories = self._topic_memories.get(topic, [])
        if not memories:
            return
        
        summary = MemorySummary.from_memories(topic, memories)
        self._summaries[topic] = summary
        
        # Keep only the most recent 5 detailed memories for this topic
        self._topic_memories[topic] = memories[-5:]
        
        # Enforce max summaries limit
        if len(self._summaries) > self.max_summaries:
            # Remove oldest summary
            oldest_topic = min(
                self._summaries.keys(),
                key=lambda t: self._summaries[t].last_updated
            )
            del self._summaries[oldest_topic]
    
    def get_topic_summary(self, topic: str) -> Optional[MemorySummary]:
        """Get the summary for a topic.

        Args:
            topic: The topic string to retrieve summary for.

        Returns:
            Optional[MemorySummary]: The summary if it exists, None otherwise.
        """
        return self._summaries.get(topic)
    
    def get_recent_memories(self, n: int = 10) -> List[MemoryItem]:
        """Get the N most recent detailed memories.

        Args:
            n: Number of recent memories to retrieve. Defaults to 10.

        Returns:
            List[MemoryItem]: List of the most recent memory items.
        """
        return list(self._detailed_memories)[-n:]
    
    def get_topic_memories(self, topic: str) -> List[MemoryItem]:
        """Get all detailed memories for a topic.

        Args:
            topic: The topic string to retrieve memories for.

        Returns:
            List[MemoryItem]: List of memory items for the topic, or empty list.
        """
        return self._topic_memories.get(topic, [])
    
    def compute_fingerprint(self) -> str:
        """
        Compute a simple hash fingerprint of memory contents.
        
        This serves as a lightweight embedding-like representation
        for comparing memory similarity between personas.
        """
        content = ""
        for memory in self._detailed_memories:
            content += f"{memory.topic}:{memory.sentiment:.1f};"
        for topic, summary in self._summaries.items():
            content += f"{topic}:{summary.average_sentiment:.1f}:{summary.interaction_count};"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_all_topics(self) -> List[str]:
        """Get all topics this memory has information about.

        Returns:
            List[str]: List of all topic strings in memory.
        """
        topics = set(self._topic_memories.keys())
        topics.update(self._summaries.keys())
        return list(topics)
    
    def clear(self) -> None:
        """Clear all memories.

        Removes all detailed memories, summaries, and topic memories.
        """
        self._detailed_memories.clear()
        self._summaries.clear()
        self._topic_memories.clear()
    
    @property
    def total_memories(self) -> int:
        """Total number of detailed memories.

        Returns:
            int: Count of detailed memories currently stored.
        """
        return len(self._detailed_memories)
    
    @property
    def total_summaries(self) -> int:
        """Total number of topic summaries.

        Returns:
            int: Count of topic summaries currently stored.
        """
        return len(self._summaries)
    
    def __repr__(self) -> str:
        """Return string representation of the Memory instance.

        Returns:
            str: String showing detailed memory and summary counts.
        """
        return f"Memory(detailed={self.total_memories}, summaries={self.total_summaries})"
