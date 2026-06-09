"""Meta-memory: knowledge about memory quality and self-assessment."""

from __future__ import annotations

import time
from typing import Any

from src.core import Memory, MemoryLayer, MemoryType


class MetaMemoryEntry:
    """A meta-memory entry that tracks quality of memory sources."""

    def __init__(self, source_id: str, source_type: str):
        self.source_id = source_id
        self.source_type = source_type
        self.reliability_score: float = 0.5
        self.usage_count: int = 0
        self.success_count: int = 0
        self.failure_count: int = 0
        self.last_evaluated: float = time.time()
        self.notes: list[str] = []

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def record_success(self) -> None:
        self.success_count += 1
        self.usage_count += 1
        self.reliability_score = self.reliability
        self.last_evaluated = time.time()

    def record_failure(self, note: str = "") -> None:
        self.failure_count += 1
        self.usage_count += 1
        self.reliability_score = self.reliability
        self.last_evaluated = time.time()
        if note:
            self.notes.append(note)


class MetaMemorySystem:
    """Layer 5: Knowledge about memory quality itself.

    Tracks which memory sources are reliable, which agents perform well,
    and which types of memories tend to be useful.

    Example meta-knowledge:
    - "Episodic memories from Tuesday sessions are often unreliable"
    - "Procedural memories for deployment have 90% success rate"
    - "Semantic agent is better than episodic for technical queries"
    """

    def __init__(self):
        self._entries: dict[str, MetaMemoryEntry] = {}
        self._retrieval_quality_history: list[dict[str, Any]] = []
        self._agent_rankings: dict[str, float] = {}

    def register_source(self, source_id: str, source_type: str) -> None:
        """Register a new memory source for tracking."""
        if source_id not in self._entries:
            self._entries[source_id] = MetaMemoryEntry(source_id, source_type)

    def record_retrieval_quality(
        self,
        query: str,
        agent_id: str,
        memory_ids: list[str],
        was_useful: bool,
        user_feedback: str = "",
    ) -> None:
        """Record the quality of a retrieval for meta-learning."""
        self._retrieval_quality_history.append({
            "query": query,
            "agent_id": agent_id,
            "memory_ids": memory_ids,
            "was_useful": was_useful,
            "feedback": user_feedback,
            "timestamp": time.time(),
        })

        # Update source reliability
        for mid in memory_ids:
            if mid in self._entries:
                if was_useful:
                    self._entries[mid].record_success()
                else:
                    self._entries[mid].record_failure(user_feedback)

        # Update agent ranking
        if agent_id not in self._agent_rankings:
            self._agent_rankings[agent_id] = 0.5
        current = self._agent_rankings[agent_id]
        self._agent_rankings[agent_id] = current * 0.9 + (1.0 if was_useful else 0.0) * 0.1

    def get_source_reliability(self, source_id: str) -> float:
        """Get reliability score for a memory source."""
        entry = self._entries.get(source_id)
        return entry.reliability_score if entry else 0.5

    def get_agent_ranking(self, agent_id: str) -> float:
        """Get performance ranking for a retrieval agent."""
        return self._agent_rankings.get(agent_id, 0.5)

    def get_unreliable_sources(self, threshold: float = 0.3) -> list[MetaMemoryEntry]:
        """Get sources that have proven unreliable."""
        return [
            entry for entry in self._entries.values()
            if entry.reliability_score < threshold and entry.usage_count >= 3
        ]

    def should_trust_memory(self, memory_id: str) -> bool:
        """Determine if a memory should be trusted based on meta-knowledge."""
        entry = self._entries.get(memory_id)
        if entry is None:
            return True  # Unknown sources get benefit of doubt
        return entry.reliability_score >= 0.4

    def generate_meta_insights(self) -> list[str]:
        """Generate meta-level insights about memory system performance."""
        insights = []

        # Agent performance insights
        if self._agent_rankings:
            best_agent = max(self._agent_rankings, key=self._agent_rankings.get)
            worst_agent = min(self._agent_rankings, key=self._agent_rankings.get)
            insights.append(
                f"Best performing agent: {best_agent} "
                f"(score: {self._agent_rankings[best_agent]:.2f})"
            )
            if self._agent_rankings[worst_agent] < 0.3:
                insights.append(
                    f"Underperforming agent: {worst_agent} — consider retraining"
                )

        # Unreliable sources
        unreliable = self.get_unreliable_sources()
        if unreliable:
            insights.append(
                f"{len(unreliable)} memory sources flagged as unreliable"
            )

        # Overall system health
        total_retrievals = len(self._retrieval_quality_history)
        if total_retrievals > 0:
            successful = sum(
                1 for r in self._retrieval_quality_history if r["was_useful"]
            )
            rate = successful / total_retrievals
            insights.append(f"Overall retrieval success rate: {rate:.1%}")

        return insights

    @property
    def tracked_sources(self) -> int:
        return len(self._entries)
