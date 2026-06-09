"""Unit tests for the memory graph with activation spreading."""

import pytest

from src.core import Memory, MemoryType
from src.graph import MemoryGraph


@pytest.fixture
def graph_with_memories():
    """Create a graph with interconnected memories."""
    graph = MemoryGraph()

    memories = [
        Memory(id="deploy_fail", content="Deployment failed", memory_type=MemoryType.EPISODIC),
        Memory(id="redis_issue", content="Redis connection timeout", memory_type=MemoryType.EPISODIC),
        Memory(id="env_vars", content="Missing environment variables", memory_type=MemoryType.SEMANTIC),
        Memory(id="docker", content="Docker compose misconfigured", memory_type=MemoryType.PROCEDURAL),
        Memory(id="unrelated", content="Meeting notes from Monday", memory_type=MemoryType.SOCIAL),
    ]

    for mem in memories:
        graph.add_memory(mem)

    # Create associations
    graph.add_association("deploy_fail", "redis_issue", weight=0.9, relation="caused_by")
    graph.add_association("redis_issue", "env_vars", weight=0.8, relation="caused_by")
    graph.add_association("env_vars", "docker", weight=0.7, relation="related_to")

    return graph


class TestMemoryGraph:
    def test_add_memory(self):
        graph = MemoryGraph()
        mem = Memory(id="test", content="test", memory_type=MemoryType.EPISODIC)
        graph.add_memory(mem)
        assert graph.node_count == 1

    def test_add_association(self):
        graph = MemoryGraph()
        m1 = Memory(id="a", content="a", memory_type=MemoryType.EPISODIC)
        m2 = Memory(id="b", content="b", memory_type=MemoryType.EPISODIC)
        graph.add_memory(m1)
        graph.add_memory(m2)
        graph.add_association("a", "b", weight=0.5)
        assert graph.edge_count == 1

    def test_activation_spreading(self, graph_with_memories):
        """Test that activation spreads from deploy_fail through the chain."""
        activations = graph_with_memories.spread_activation(
            seed_ids=["deploy_fail"],
            initial_activation=1.0,
            decay_factor=0.7,
        )

        assert "deploy_fail" in activations
        assert "redis_issue" in activations
        assert "env_vars" in activations
        # Unrelated memory should NOT be activated
        assert "unrelated" not in activations

    def test_activation_decays_with_distance(self, graph_with_memories):
        activations = graph_with_memories.spread_activation(
            seed_ids=["deploy_fail"],
            initial_activation=1.0,
            decay_factor=0.6,
        )

        # Closer memories should have higher activation
        assert activations.get("redis_issue", 0) > activations.get("env_vars", 0)
        assert activations.get("env_vars", 0) > activations.get("docker", 0)

    def test_get_activated_memories(self, graph_with_memories):
        results = graph_with_memories.get_activated_memories(
            seed_ids=["deploy_fail"], top_k=3
        )
        assert len(results) > 0
        # Results should be (Memory, activation) tuples
        for mem, activation in results:
            assert isinstance(mem, Memory)
            assert 0 < activation <= 1.0

    def test_reinforce_edge(self, graph_with_memories):
        graph_with_memories.reinforce_edge("deploy_fail", "redis_issue", amount=0.5)
        # Verify via activation spreading - should be stronger now
        activations = graph_with_memories.spread_activation(["deploy_fail"])
        assert activations.get("redis_issue", 0) > 0.5

    def test_find_path(self, graph_with_memories):
        path = graph_with_memories.find_path("deploy_fail", "docker")
        assert path is not None
        assert path[0] == "deploy_fail"
        assert path[-1] == "docker"

    def test_no_path(self, graph_with_memories):
        path = graph_with_memories.find_path("unrelated", "docker")
        assert path is None
