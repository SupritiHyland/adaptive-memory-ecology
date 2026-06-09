"""Main LangGraph orchestration for the Adaptive Memory Ecology."""

from __future__ import annotations

import asyncio
from typing import Any, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.core import AgentState, Memory, MemoryLayer, MemoryType, RetrievalResult
from src.core.llm import get_llm
from src.memory import EpisodicMemoryStore, ShortTermBuffer
from src.graph import MemoryGraph
from src.agents import EpisodicAgent, SemanticAgent, ProceduralAgent, SocialAgent
from src.retrieval import RetrievalArbitrator
from src.consolidation import ConsolidationEngine
from src.evolution import AntColonyWorkflowOptimizer
from src.evolution.strategies import EvolutionaryOptimizer
from src.swarm import SwarmCoordinator
from src.meta import MetaMemorySystem


class MemoryEcology:
    """The complete adaptive memory ecology system.

    Orchestrates all components:
    - Hierarchical memory stores
    - Specialized retrieval agents
    - Competitive retrieval arbitration
    - Swarm processing
    - Consolidation cycles
    - Evolutionary optimization
    - Meta-memory tracking
    """

    def __init__(self):
        # Core stores
        self.short_term = ShortTermBuffer(max_size=50)
        self.episodic_store = EpisodicMemoryStore()
        self.memory_graph = MemoryGraph()

        # Agents
        self.episodic_agent = EpisodicAgent(self.episodic_store)
        self.semantic_agent = SemanticAgent(self.episodic_store)
        self.procedural_agent = ProceduralAgent(AntColonyWorkflowOptimizer())
        self.social_agent = SocialAgent(self.episodic_store)

        # Orchestration
        self.arbitrator = RetrievalArbitrator([
            self.episodic_agent,
            self.semantic_agent,
            self.procedural_agent,
            self.social_agent,
        ])

        # Swarm
        self.swarm = SwarmCoordinator()

        # Consolidation
        self.consolidation = ConsolidationEngine(
            self.episodic_store, self.memory_graph
        )

        # Evolution
        self.optimizer = EvolutionaryOptimizer()
        self.ant_colony = AntColonyWorkflowOptimizer()

        # Meta-memory
        self.meta_memory = MetaMemorySystem()

        # LLM
        self._llm = get_llm()

        # Build the graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("swarm_process", self._swarm_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("feedback", self._feedback_node)

        # Define edges
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "retrieve")
        workflow.add_edge("retrieve", "swarm_process")
        workflow.add_edge("swarm_process", "generate")
        workflow.add_edge("generate", "feedback")
        workflow.add_edge("feedback", END)

        return workflow.compile()

    async def _ingest_node(self, state: AgentState) -> dict[str, Any]:
        """Ingest the query into short-term memory."""
        memory = Memory(
            content=state.query,
            memory_type=MemoryType.EPISODIC,
            layer=MemoryLayer.RAW_EPISODIC,
            salience=0.6,
        )

        # Add to short-term buffer
        overflow = self.short_term.add(memory)

        # If overflow, persist to episodic store
        if overflow:
            await self.episodic_store.store(overflow)
            self.memory_graph.add_memory(overflow)

        return {"current_step": "retrieve", "context": [memory]}

    async def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Run competitive retrieval across all agents."""
        strategy = self.optimizer.get_active_strategy()

        results = await self.arbitrator.retrieve(
            query=state.query,
            context={"strategy": strategy},
            top_k=10,
            relevance_threshold=strategy.activation_threshold,
        )

        return {"retrieved_memories": results, "current_step": "swarm_process"}

    async def _swarm_node(self, state: AgentState) -> dict[str, Any]:
        """Process retrieved memories through the swarm."""
        memories = [r.memory for r in state.retrieved_memories]

        processed = await self.swarm.process(memories, state.query)

        # Update retrieved results with swarm-processed memories
        updated_results = []
        for mem in processed:
            updated_results.append(RetrievalResult(
                memory=mem,
                relevance_score=mem.salience,
                confidence=mem.activation_level,
                agent_id="swarm",
            ))

        return {"retrieved_memories": updated_results, "current_step": "generate"}

    async def _generate_node(self, state: AgentState) -> dict[str, Any]:
        """Generate response using retrieved context."""
        context_str = "\n".join(
            f"[{r.memory.memory_type.value}] {r.memory.content}"
            for r in state.retrieved_memories[:5]
        )

        prompt = (
            f"Context from memory:\n{context_str}\n\n"
            f"Query: {state.query}\n\n"
            "Respond based on the available memory context. "
            "If no relevant context exists, say so."
        )

        response = await self._llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        return {"response": response_text, "current_step": "feedback"}

    async def _feedback_node(self, state: AgentState) -> dict[str, Any]:
        """Record meta-memory feedback."""
        # Track which agents contributed
        for result in state.retrieved_memories:
            self.meta_memory.register_source(result.memory.id, result.agent_id)

        return {"current_step": "done"}

    async def query(self, user_query: str) -> str:
        """Main entry point: query the memory ecology."""
        initial_state = AgentState(query=user_query)
        result = await self._graph.ainvoke(initial_state)
        return result.get("response", "No response generated.")

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a new memory to the ecology."""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
        )
        memory_id = await self.episodic_store.store(memory)
        self.memory_graph.add_memory(memory)
        return memory_id

    async def consolidate(self) -> dict[str, int]:
        """Run a consolidation/sleep cycle."""
        stats = await self.consolidation.run_consolidation_cycle()
        self.ant_colony.evaporate()
        self.optimizer.evolve()
        return stats

    def provide_feedback(self, was_useful: bool, agent_id: str = "") -> None:
        """Provide feedback on last retrieval quality."""
        if agent_id:
            self.arbitrator.record_feedback(agent_id, was_useful)
        # Update evolutionary optimizer
        strategy = self.optimizer.get_active_strategy()
        self.optimizer.evaluate_strategy(
            strategy.id,
            task_success=was_useful,
        )

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health metrics."""
        return {
            "episodic_store_size": self.episodic_store.size,
            "short_term_buffer": self.short_term.size,
            "graph_nodes": self.memory_graph.node_count,
            "graph_edges": self.memory_graph.edge_count,
            "swarm_agents": self.swarm.agent_count,
            "meta_sources_tracked": self.meta_memory.tracked_sources,
            "evolution_generation": self.optimizer.generation,
            "evolution_stats": self.optimizer.population_stats,
            "consolidation_cycles": self.consolidation.consolidation_count,
            "meta_insights": self.meta_memory.generate_meta_insights(),
        }
