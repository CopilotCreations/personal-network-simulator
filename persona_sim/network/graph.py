"""
Social graph for persona network simulation.

Defines who can communicate with whom and the strength
of connections between personas.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Iterator
from enum import Enum
import random
import math


class ConnectionType(Enum):
    """Types of connections between personas."""
    FOLLOWER = "follower"  # One-way: A follows B
    MUTUAL = "mutual"  # Two-way: A and B follow each other
    CLOSE = "close"  # Strong mutual connection
    ACQUAINTANCE = "acquaintance"  # Weak connection


@dataclass
class Connection:
    """
    A directed connection between two personas.
    
    Connections have strength and type, which affect
    how information flows between personas.
    """
    source_id: str
    target_id: str
    connection_type: ConnectionType = ConnectionType.FOLLOWER
    strength: float = 0.5  # 0.0 to 1.0
    interaction_count: int = 0
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
    
    def record_interaction(self) -> None:
        """Record an interaction and slightly strengthen the connection."""
        self.interaction_count += 1
        self.strength = min(1.0, self.strength + 0.01)
    
    def decay(self, factor: float = 0.99) -> None:
        """Decay connection strength over time."""
        self.strength *= factor


class SocialGraph:
    """
    A directed graph representing social connections between personas.
    
    Supports:
    - Adding/removing personas and connections
    - Querying neighbors and paths
    - Graph metrics (density, clustering)
    - Community detection (simplified)
    """
    
    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[Tuple[str, str], Connection] = {}
        self._outgoing: Dict[str, Set[str]] = {}
        self._incoming: Dict[str, Set[str]] = {}
    
    def add_node(self, persona_id: str) -> None:
        """Add a persona to the graph."""
        if persona_id not in self._nodes:
            self._nodes.add(persona_id)
            self._outgoing[persona_id] = set()
            self._incoming[persona_id] = set()
    
    def remove_node(self, persona_id: str) -> None:
        """Remove a persona and all their connections."""
        if persona_id not in self._nodes:
            return
        
        # Remove all outgoing connections
        for target_id in list(self._outgoing.get(persona_id, [])):
            self.remove_connection(persona_id, target_id)
        
        # Remove all incoming connections
        for source_id in list(self._incoming.get(persona_id, [])):
            self.remove_connection(source_id, persona_id)
        
        self._nodes.discard(persona_id)
        self._outgoing.pop(persona_id, None)
        self._incoming.pop(persona_id, None)
    
    def add_connection(
        self,
        source_id: str,
        target_id: str,
        connection_type: ConnectionType = ConnectionType.FOLLOWER,
        strength: float = 0.5,
    ) -> Connection:
        """Add a directed connection between two personas."""
        # Ensure both nodes exist
        self.add_node(source_id)
        self.add_node(target_id)
        
        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            strength=strength,
        )
        
        self._edges[(source_id, target_id)] = connection
        self._outgoing[source_id].add(target_id)
        self._incoming[target_id].add(source_id)
        
        # For mutual connections, add the reverse edge
        if connection_type in (ConnectionType.MUTUAL, ConnectionType.CLOSE):
            reverse = Connection(
                source_id=target_id,
                target_id=source_id,
                connection_type=connection_type,
                strength=strength,
            )
            self._edges[(target_id, source_id)] = reverse
            self._outgoing[target_id].add(source_id)
            self._incoming[source_id].add(target_id)
        
        return connection
    
    def remove_connection(self, source_id: str, target_id: str) -> None:
        """Remove a connection between two personas."""
        key = (source_id, target_id)
        if key in self._edges:
            del self._edges[key]
            self._outgoing[source_id].discard(target_id)
            self._incoming[target_id].discard(source_id)
    
    def get_connection(self, source_id: str, target_id: str) -> Optional[Connection]:
        """Get the connection between two personas."""
        return self._edges.get((source_id, target_id))
    
    def get_followers(self, persona_id: str) -> List[str]:
        """Get all personas that follow this persona."""
        return list(self._incoming.get(persona_id, []))
    
    def get_following(self, persona_id: str) -> List[str]:
        """Get all personas this persona follows."""
        return list(self._outgoing.get(persona_id, []))
    
    def get_mutual_connections(self, persona_id: str) -> List[str]:
        """Get personas with mutual connections."""
        following = self._outgoing.get(persona_id, set())
        followers = self._incoming.get(persona_id, set())
        return list(following & followers)
    
    def get_neighbors(self, persona_id: str) -> List[str]:
        """Get all connected personas (in or out)."""
        following = self._outgoing.get(persona_id, set())
        followers = self._incoming.get(persona_id, set())
        return list(following | followers)
    
    @property
    def nodes(self) -> List[str]:
        """All persona IDs in the graph."""
        return list(self._nodes)
    
    @property
    def edges(self) -> List[Connection]:
        """All connections in the graph."""
        return list(self._edges.values())
    
    @property
    def node_count(self) -> int:
        """Number of personas in the graph."""
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        """Number of connections in the graph."""
        return len(self._edges)
    
    def density(self) -> float:
        """
        Calculate graph density.
        
        Density = edges / (nodes * (nodes - 1))
        For directed graphs.
        """
        n = self.node_count
        if n < 2:
            return 0.0
        max_edges = n * (n - 1)
        return self.edge_count / max_edges
    
    def clustering_coefficient(self, persona_id: str) -> float:
        """
        Calculate local clustering coefficient for a persona.
        
        Measures how interconnected a persona's neighbors are.
        """
        neighbors = self.get_neighbors(persona_id)
        k = len(neighbors)
        
        if k < 2:
            return 0.0
        
        # Count edges between neighbors
        edges_between = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if (n1, n2) in self._edges or (n2, n1) in self._edges:
                    edges_between += 1
        
        max_edges = k * (k - 1) / 2
        return edges_between / max_edges
    
    def average_clustering(self) -> float:
        """Calculate average clustering coefficient across all personas."""
        if self.node_count == 0:
            return 0.0
        
        total = sum(self.clustering_coefficient(n) for n in self._nodes)
        return total / self.node_count
    
    def shortest_path_length(self, source_id: str, target_id: str) -> Optional[int]:
        """
        Find shortest path length between two personas using BFS.
        
        Returns None if no path exists.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        if source_id == target_id:
            return 0
        
        visited = {source_id}
        queue = [(source_id, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            for neighbor in self._outgoing.get(current, []):
                if neighbor == target_id:
                    return distance + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return None
    
    def find_communities(self, min_size: int = 2) -> List[Set[str]]:
        """
        Find communities using simple connected components.
        
        For more sophisticated detection, use proper algorithms
        like Louvain or label propagation.
        """
        visited = set()
        communities = []
        
        for node in self._nodes:
            if node in visited:
                continue
            
            # BFS to find connected component
            component = set()
            queue = [node]
            
            while queue:
                current = queue.pop(0)
                if current in component:
                    continue
                
                component.add(current)
                visited.add(current)
                
                # Add all neighbors
                queue.extend(
                    n for n in self.get_neighbors(current)
                    if n not in component
                )
            
            if len(component) >= min_size:
                communities.append(component)
        
        return communities
    
    def decay_all_connections(self, factor: float = 0.99) -> None:
        """Apply decay to all connection strengths."""
        for connection in self._edges.values():
            connection.decay(factor)
    
    @classmethod
    def create_random(
        cls,
        persona_ids: List[str],
        connection_probability: float = 0.3,
        seed: Optional[int] = None,
    ) -> "SocialGraph":
        """
        Create a random graph with given connection probability.
        
        Uses Erdős–Rényi model.
        """
        rng = random.Random(seed)
        graph = cls()
        
        for persona_id in persona_ids:
            graph.add_node(persona_id)
        
        for i, source in enumerate(persona_ids):
            for target in persona_ids[i + 1:]:
                if rng.random() < connection_probability:
                    # Randomly choose connection type
                    if rng.random() < 0.5:
                        graph.add_connection(source, target, ConnectionType.MUTUAL)
                    else:
                        graph.add_connection(source, target, ConnectionType.FOLLOWER)
        
        return graph
    
    @classmethod
    def create_small_world(
        cls,
        persona_ids: List[str],
        k: int = 4,
        rewire_prob: float = 0.1,
        seed: Optional[int] = None,
    ) -> "SocialGraph":
        """
        Create a small-world graph (Watts-Strogatz model).
        
        Args:
            persona_ids: List of persona IDs
            k: Each node connected to k nearest neighbors
            rewire_prob: Probability of rewiring each edge
        """
        rng = random.Random(seed)
        graph = cls()
        n = len(persona_ids)
        
        for persona_id in persona_ids:
            graph.add_node(persona_id)
        
        # Create ring lattice
        for i, source in enumerate(persona_ids):
            for j in range(1, k // 2 + 1):
                target_idx = (i + j) % n
                target = persona_ids[target_idx]
                graph.add_connection(source, target, ConnectionType.MUTUAL)
        
        # Rewire edges
        for source_id in persona_ids:
            for target_id in list(graph.get_following(source_id)):
                if rng.random() < rewire_prob:
                    # Remove this edge and add a random one
                    graph.remove_connection(source_id, target_id)
                    graph.remove_connection(target_id, source_id)
                    
                    # Pick a random new target
                    candidates = [p for p in persona_ids 
                                  if p != source_id and p not in graph.get_following(source_id)]
                    if candidates:
                        new_target = rng.choice(candidates)
                        graph.add_connection(source_id, new_target, ConnectionType.MUTUAL)
        
        return graph
    
    def __repr__(self) -> str:
        return f"SocialGraph(nodes={self.node_count}, edges={self.edge_count})"
