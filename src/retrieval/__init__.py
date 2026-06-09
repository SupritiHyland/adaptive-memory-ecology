"""Retrieval competition and attention arbitration."""

from __future__ import annotations

import asyncio
from typing import Any

from src.core import RetrievalResult
from src.agents import BaseMemoryAgent


class RetrievalArbitrator:
    """Meta-agent that orchestrates retrieval competition.

    Implements immune-system-inspired agent competition:
    1. All agents bid on relevance
    2. Top agents retrieve memories
    3. Results are merged with conflict resolution
    """

    def __init__(self, agents: list[BaseMemoryAgent], max_concurrent: int = 4):
        self._agents = agents
        self._max_concurrent = max_concurrent
        self._agent_performance: dict[str, list[bool]] = {}

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        top_k: int = 10,
        relevance_threshold: float = 0.3,
    ) -> list[RetrievalResult]:
        """Run competitive retrieval across all agents."""
        context = context or {}

        # Phase 1: Relevance bidding
        relevance_scores = await self._compute_relevance_bids(query)

        # Phase 2: Select top agents
        active_agents = [
            (agent, score)
            for agent, score in relevance_scores
            if score >= relevance_threshold
        ]
        active_agents.sort(key=lambda x: x[1], reverse=True)
        active_agents = active_agents[:self._max_concurrent]

        if not active_agents:
            # Fallback: use all agents
            active_agents = relevance_scores[:self._max_concurrent]

        # Phase 3: Parallel retrieval
        tasks = [agent.retrieve(query, context) for agent, _ in active_agents]
        results_per_agent = await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 4: Merge and deduplicate
        all_results: list[RetrievalResult] = []
        for results in results_per_agent:
            if isinstance(results, Exception):
                continue
            all_results.extend(results)

        # Phase 5: Score and rank
        ranked = self._rank_results(all_results, relevance_scores)
        return ranked[:top_k]

    async def _compute_relevance_bids(
        self, query: str
    ) -> list[tuple[BaseMemoryAgent, float]]:
        """Have all agents bid on their relevance to the query."""
        tasks = [agent.compute_relevance(query) for agent in self._agents]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for agent, score in zip(self._agents, scores):
            if isinstance(score, Exception):
                score = 0.1
            # Weight by historical performance
            perf_weight = self._get_performance_weight(agent.agent_id)
            results.append((agent, score * perf_weight))

        return results

    def _rank_results(
        self,
        results: list[RetrievalResult],
        agent_scores: list[tuple[BaseMemoryAgent, float]],
    ) -> list[RetrievalResult]:
        """Rank results using weighted ensemble scoring."""
        agent_weight_map = {agent.agent_id: score for agent, score in agent_scores}

        # Composite score: relevance * confidence * agent_weight + novelty_bonus
        def composite_score(r: RetrievalResult) -> float:
            agent_weight = agent_weight_map.get(r.agent_id, 0.5)
            base = r.relevance_score * r.confidence * agent_weight
            novelty_bonus = r.novelty * 0.1
            return base + novelty_bonus

        results.sort(key=composite_score, reverse=True)

        # Deduplicate by memory id
        seen_ids: set[str] = set()
        unique_results = []
        for r in results:
            if r.memory.id not in seen_ids:
                seen_ids.add(r.memory.id)
                unique_results.append(r)

        return unique_results

    def _get_performance_weight(self, agent_id: str) -> float:
        """Get performance-based weight for an agent."""
        history = self._agent_performance.get(agent_id, [])
        if not history:
            return 1.0
        recent = history[-20:]
        success_rate = sum(recent) / len(recent)
        return 0.5 + success_rate * 0.5  # Range: 0.5 to 1.0

    def record_feedback(self, agent_id: str, was_useful: bool) -> None:
        """Record feedback for agent performance tracking."""
        if agent_id not in self._agent_performance:
            self._agent_performance[agent_id] = []
        self._agent_performance[agent_id].append(was_useful)

        # Also update the agent directly
        for agent in self._agents:
            if agent.agent_id == agent_id:
                agent.record_feedback(was_useful)
                break
