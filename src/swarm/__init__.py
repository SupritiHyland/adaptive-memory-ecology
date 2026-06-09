"""Swarm retrieval agents - many weak specialized agents working together."""

from __future__ import annotations

import asyncio
from typing import Any

from src.core import Memory, MemoryType, RetrievalResult
from src.memory import EpisodicMemoryStore


class SwarmAgent:
    """A lightweight specialized swarm agent."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self._active = True

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Process memories according to specialization."""
        return memories


class SalienceScorer(SwarmAgent):
    """Scores memories by salience/importance."""

    def __init__(self):
        super().__init__("salience_scorer", "scoring")

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Re-score memories based on query-specific salience."""
        for mem in memories:
            # Boost salience if memory type matches query intent
            if any(word in query.lower() for word in mem.content.lower().split()[:5]):
                mem.salience = min(1.0, mem.salience + 0.2)
        return sorted(memories, key=lambda m: m.salience, reverse=True)


class ContradictionDetector(SwarmAgent):
    """Detects contradictions between memories."""

    def __init__(self):
        super().__init__("contradiction_detector", "validation")

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Flag memories that might contradict each other."""
        # Simple heuristic: mark memories with conflicting metadata
        seen_topics: dict[str, Memory] = {}
        for mem in memories:
            topic = mem.metadata.get("topic", "")
            if topic and topic in seen_topics:
                mem.metadata["potential_contradiction"] = True
                mem.metadata["contradicts_with"] = seen_topics[topic].id
            elif topic:
                seen_topics[topic] = mem
        return memories


class TemporalDecayManager(SwarmAgent):
    """Manages time-based memory decay."""

    def __init__(self, decay_rate: float = 0.05):
        super().__init__("temporal_decay", "maintenance")
        self._decay_rate = decay_rate

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Apply temporal decay and filter stale memories."""
        for mem in memories:
            mem.decay(rate=self._decay_rate)
        # Filter out nearly-dead memories
        return [m for m in memories if m.activation_level > 0.05]


class MemoryCleaner(SwarmAgent):
    """Removes duplicate and low-quality memories."""

    def __init__(self):
        super().__init__("memory_cleaner", "maintenance")

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Deduplicate and clean memories."""
        seen_content: set[str] = set()
        cleaned = []
        for mem in memories:
            content_key = mem.content[:100].lower().strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                cleaned.append(mem)
        return cleaned


class SwarmCoordinator:
    """Coordinates the swarm of specialized agents.

    Implements emergent behavior through parallel processing:
    each agent independently processes memories, then results merge.
    """

    def __init__(self):
        self._agents: list[SwarmAgent] = [
            SalienceScorer(),
            ContradictionDetector(),
            TemporalDecayManager(),
            MemoryCleaner(),
        ]

    def add_agent(self, agent: SwarmAgent) -> None:
        """Add a new swarm agent."""
        self._agents.append(agent)

    async def process(self, memories: list[Memory], query: str) -> list[Memory]:
        """Run all swarm agents in parallel pipeline."""
        # Phase 1: Maintenance agents (clean + decay)
        maintenance_agents = [a for a in self._agents if a.role == "maintenance"]
        current = memories
        for agent in maintenance_agents:
            current = await agent.process(current, query)

        # Phase 2: Validation agents (parallel)
        validation_agents = [a for a in self._agents if a.role == "validation"]
        if validation_agents:
            tasks = [agent.process(current.copy(), query) for agent in validation_agents]
            results = await asyncio.gather(*tasks)
            # Merge validation metadata
            for result_set in results:
                for i, mem in enumerate(result_set):
                    if i < len(current):
                        current[i].metadata.update(mem.metadata)

        # Phase 3: Scoring agents
        scoring_agents = [a for a in self._agents if a.role == "scoring"]
        for agent in scoring_agents:
            current = await agent.process(current, query)

        return current

    @property
    def agent_count(self) -> int:
        return len(self._agents)
