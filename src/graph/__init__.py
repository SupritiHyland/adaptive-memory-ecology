"""Graph-based memory using NetworkX for activation spreading."""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.core import Memory, MemoryType


class MemoryGraph:
    """Graph memory with activation spreading.

    Implements spiking/activation-based retrieval where memories
    activate related memories through association links.
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_memory(self, memory: Memory) -> None:
        """Add a memory node to the graph."""
        self._graph.add_node(
            memory.id,
            memory=memory,
            activation=memory.activation_level,
            memory_type=memory.memory_type.value,
        )

    def add_association(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        relation: str = "related_to",
    ) -> None:
        """Create a weighted edge between memories."""
        if source_id in self._graph and target_id in self._graph:
            self._graph.add_edge(
                source_id,
                target_id,
                weight=weight,
                relation=relation,
            )

    def spread_activation(
        self,
        seed_ids: list[str],
        initial_activation: float = 1.0,
        decay_factor: float = 0.6,
        max_depth: int = 3,
        threshold: float = 0.1,
    ) -> dict[str, float]:
        """Spread activation from seed nodes through the graph.

        Returns dict of node_id -> activation level.
        """
        activations: dict[str, float] = {}

        # Initialize seeds
        frontier = [(nid, initial_activation) for nid in seed_ids if nid in self._graph]

        visited_depth: dict[str, int] = {}

        for depth in range(max_depth):
            next_frontier = []
            for node_id, activation in frontier:
                if activation < threshold:
                    continue

                # Accumulate activation
                current = activations.get(node_id, 0.0)
                activations[node_id] = min(1.0, current + activation)

                if node_id in visited_depth and visited_depth[node_id] <= depth:
                    continue
                visited_depth[node_id] = depth

                # Spread to neighbors
                for neighbor in self._graph.successors(node_id):
                    edge_weight = self._graph[node_id][neighbor].get("weight", 1.0)
                    spread = activation * decay_factor * edge_weight
                    if spread >= threshold:
                        next_frontier.append((neighbor, spread))

            frontier = next_frontier

        return activations

    def get_activated_memories(
        self,
        seed_ids: list[str],
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Memory, float]]:
        """Get top-k memories by activation spreading."""
        activations = self.spread_activation(seed_ids, **kwargs)

        # Sort by activation, exclude seeds
        sorted_activations = sorted(
            [(nid, act) for nid, act in activations.items() if nid not in seed_ids],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for node_id, activation in sorted_activations:
            memory = self._graph.nodes[node_id].get("memory")
            if memory:
                results.append((memory, activation))

        return results

    def find_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Find shortest path between two memories."""
        try:
            return nx.shortest_path(self._graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_clusters(self) -> list[set[str]]:
        """Find memory clusters using community detection."""
        undirected = self._graph.to_undirected()
        if len(undirected) == 0:
            return []
        components = list(nx.connected_components(undirected))
        return [comp for comp in components if len(comp) > 1]

    def reinforce_edge(self, source_id: str, target_id: str, amount: float = 0.1) -> None:
        """Strengthen association between memories."""
        if self._graph.has_edge(source_id, target_id):
            current = self._graph[source_id][target_id].get("weight", 1.0)
            self._graph[source_id][target_id]["weight"] = min(5.0, current + amount)

    def weaken_edge(self, source_id: str, target_id: str, amount: float = 0.1) -> None:
        """Weaken association between memories."""
        if self._graph.has_edge(source_id, target_id):
            current = self._graph[source_id][target_id].get("weight", 1.0)
            new_weight = current - amount
            if new_weight <= 0:
                self._graph.remove_edge(source_id, target_id)
            else:
                self._graph[source_id][target_id]["weight"] = new_weight

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()
