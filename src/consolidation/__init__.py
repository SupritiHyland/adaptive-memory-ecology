"""Memory consolidation - sleep cycles, hierarchical compression, homeostasis."""

from __future__ import annotations

import time
from typing import Any

from src.core import Memory, MemoryLayer, MemoryType
from src.core.llm import get_llm
from src.memory import EpisodicMemoryStore, ShortTermBuffer
from src.graph import MemoryGraph


class ConsolidationEngine:
    """Implements sleep/consolidation cycles for memory homeostasis.

    Biological analogy: sleep consolidation in the hippocampus-cortex system.
    - Compresses raw episodes into summaries
    - Extracts semantic concepts
    - Builds procedural policies
    - Prunes low-salience memories
    """

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        memory_graph: MemoryGraph,
        decay_rate: float = 0.05,
        prune_threshold: float = 0.05,
    ):
        self._episodic_store = episodic_store
        self._memory_graph = memory_graph
        self._decay_rate = decay_rate
        self._prune_threshold = prune_threshold
        self._llm = get_llm()
        self._consolidation_count = 0

    async def run_consolidation_cycle(self) -> dict[str, int]:
        """Run a full consolidation cycle (a 'sleep' cycle).

        Returns stats about what was consolidated.
        """
        stats = {
            "decayed": 0,
            "pruned": 0,
            "summarized": 0,
            "concepts_extracted": 0,
        }

        # Step 1: Apply decay
        self._episodic_store.apply_decay()
        stats["decayed"] = self._episodic_store.size

        # Step 2: Prune dead memories
        stats["pruned"] = self._prune_memories()

        # Step 3: Compress episodes into summaries
        stats["summarized"] = await self._compress_episodes()

        # Step 4: Extract concepts
        stats["concepts_extracted"] = await self._extract_concepts()

        self._consolidation_count += 1
        return stats

    def _prune_memories(self) -> int:
        """Remove memories below activation threshold (aggressive forgetting)."""
        pruned = 0
        for memory_id in list(self._episodic_store._memories.keys()):
            memory = self._episodic_store._memories[memory_id]
            if memory.activation_level < self._prune_threshold:
                del self._episodic_store._memories[memory_id]
                pruned += 1
        return pruned

    async def _compress_episodes(self, batch_size: int = 10) -> int:
        """Compress groups of related episodic memories into summaries."""
        # Get clusters from the memory graph
        clusters = self._memory_graph.get_clusters()
        compressed = 0

        for cluster_ids in clusters[:5]:  # Process top 5 clusters per cycle
            cluster_memories = [
                self._episodic_store.get_memory(mid)
                for mid in cluster_ids
                if self._episodic_store.get_memory(mid) is not None
            ]

            if len(cluster_memories) < 3:
                continue

            # Use LLM to summarize
            contents = "\n".join(m.content for m in cluster_memories if m)
            prompt = (
                "Summarize the following related memories into a single concise "
                "knowledge statement:\n\n"
                f"{contents}\n\n"
                "Summary:"
            )

            response = await self._llm.ainvoke(prompt)
            summary_text = response.content if hasattr(response, "content") else str(response)

            # Create summary memory (Layer 2)
            summary_memory = Memory(
                content=summary_text,
                memory_type=MemoryType.SEMANTIC,
                layer=MemoryLayer.SUMMARY,
                salience=0.7,
                source_memory_ids=[m.id for m in cluster_memories if m],
            )
            await self._episodic_store.store(summary_memory)
            self._memory_graph.add_memory(summary_memory)

            # Link summary to source memories
            for mem in cluster_memories:
                if mem:
                    self._memory_graph.add_association(
                        summary_memory.id, mem.id, weight=0.8, relation="summarizes"
                    )

            compressed += 1

        return compressed

    async def _extract_concepts(self) -> int:
        """Extract semantic concepts from summaries (Layer 2 → Layer 3)."""
        # Find summary-layer memories with high salience
        summaries = [
            m for m in self._episodic_store._memories.values()
            if m.layer == MemoryLayer.SUMMARY and m.salience > 0.5
        ]

        if len(summaries) < 2:
            return 0

        contents = "\n".join(m.content for m in summaries[:10])
        prompt = (
            "Extract 2-3 key concepts or principles from these summaries. "
            "Each concept should be a single clear statement:\n\n"
            f"{contents}\n\n"
            "Concepts (one per line):"
        )

        response = await self._llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        concepts = [c.strip() for c in response_text.strip().split("\n") if c.strip()]

        for concept_text in concepts:
            concept_memory = Memory(
                content=concept_text,
                memory_type=MemoryType.SEMANTIC,
                layer=MemoryLayer.SEMANTIC_CONCEPT,
                salience=0.8,
            )
            await self._episodic_store.store(concept_memory)
            self._memory_graph.add_memory(concept_memory)

        return len(concepts)

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count
