"""Specialized retrieval agents that compete for attention."""

from __future__ import annotations

import abc
from typing import Any

from src.core import Memory, MemoryType, RetrievalResult


class BaseMemoryAgent(abc.ABC):
    """Base class for specialized memory retrieval agents.

    Each agent acts like an immune system cell — specialized in detecting
    and retrieving specific types of relevant memories.
    """

    def __init__(self, agent_id: str, specialization: MemoryType):
        self.agent_id = agent_id
        self.specialization = specialization
        self._confidence_history: list[float] = []

    @abc.abstractmethod
    async def retrieve(self, query: str, context: dict[str, Any]) -> list[RetrievalResult]:
        """Retrieve relevant memories for the query."""
        ...

    @abc.abstractmethod
    async def compute_relevance(self, query: str) -> float:
        """Compute how relevant this agent is for the query (0-1)."""
        ...

    @property
    def average_confidence(self) -> float:
        if not self._confidence_history:
            return 0.5
        return sum(self._confidence_history[-20:]) / len(self._confidence_history[-20:])

    def record_feedback(self, was_useful: bool) -> None:
        """Record whether the retrieval was useful."""
        self._confidence_history.append(1.0 if was_useful else 0.0)


class EpisodicAgent(BaseMemoryAgent):
    """Retrieves recent experiences and conversations."""

    def __init__(self, episodic_store):
        super().__init__("episodic_agent", MemoryType.EPISODIC)
        self._store = episodic_store

    async def retrieve(self, query: str, context: dict[str, Any]) -> list[RetrievalResult]:
        memories = await self._store.retrieve(query, n_results=5)
        return [
            RetrievalResult(
                memory=mem,
                relevance_score=mem.activation_level,
                confidence=self.average_confidence,
                novelty=1.0 - (mem.access_count / max(mem.access_count + 1, 10)),
                agent_id=self.agent_id,
            )
            for mem in memories
        ]

    async def compute_relevance(self, query: str) -> float:
        # Episodic agent is always somewhat relevant
        return 0.6


class SemanticAgent(BaseMemoryAgent):
    """Retrieves conceptual knowledge and facts."""

    def __init__(self, episodic_store):
        super().__init__("semantic_agent", MemoryType.SEMANTIC)
        self._store = episodic_store

    async def retrieve(self, query: str, context: dict[str, Any]) -> list[RetrievalResult]:
        memories = await self._store.retrieve(query, n_results=5)
        # Filter to semantic layer memories
        semantic_memories = [m for m in memories if m.layer.value >= 3]
        return [
            RetrievalResult(
                memory=mem,
                relevance_score=mem.salience,
                confidence=self.average_confidence,
                novelty=0.3,
                agent_id=self.agent_id,
            )
            for mem in semantic_memories
        ]

    async def compute_relevance(self, query: str) -> float:
        knowledge_indicators = ["what is", "how does", "explain", "define", "why"]
        return 0.8 if any(ind in query.lower() for ind in knowledge_indicators) else 0.4


class ProceduralAgent(BaseMemoryAgent):
    """Retrieves workflow knowledge and procedures."""

    def __init__(self, workflow_store):
        super().__init__("procedural_agent", MemoryType.PROCEDURAL)
        self._workflow_store = workflow_store

    async def retrieve(self, query: str, context: dict[str, Any]) -> list[RetrievalResult]:
        # Retrieve best workflows for the task
        workflows = self._workflow_store.get_best_workflows(query, top_k=3)
        results = []
        for workflow, score in workflows:
            memory = Memory(
                content=f"Workflow: {' → '.join(s.tool for s in workflow.steps)}",
                memory_type=MemoryType.PROCEDURAL,
                metadata={"workflow_id": workflow.id, "success_rate": workflow.success_rate},
                salience=workflow.pheromone_strength / 10.0,
            )
            results.append(RetrievalResult(
                memory=memory,
                relevance_score=score,
                confidence=workflow.success_rate,
                novelty=0.2,
                agent_id=self.agent_id,
            ))
        return results

    async def compute_relevance(self, query: str) -> float:
        action_indicators = ["how to", "steps", "process", "workflow", "do", "build", "create"]
        return 0.9 if any(ind in query.lower() for ind in action_indicators) else 0.3


class SocialAgent(BaseMemoryAgent):
    """Retrieves user preferences and interaction patterns."""

    def __init__(self, episodic_store):
        super().__init__("social_agent", MemoryType.SOCIAL)
        self._store = episodic_store

    async def retrieve(self, query: str, context: dict[str, Any]) -> list[RetrievalResult]:
        memories = await self._store.retrieve(query, n_results=3)
        social_memories = [m for m in memories if m.memory_type == MemoryType.SOCIAL]
        return [
            RetrievalResult(
                memory=mem,
                relevance_score=mem.salience,
                confidence=self.average_confidence,
                novelty=0.4,
                agent_id=self.agent_id,
            )
            for mem in social_memories
        ]

    async def compute_relevance(self, query: str) -> float:
        social_indicators = ["prefer", "like", "style", "user", "always", "never"]
        return 0.7 if any(ind in query.lower() for ind in social_indicators) else 0.2
