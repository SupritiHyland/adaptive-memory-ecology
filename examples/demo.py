"""Example: Demonstrating the Memory Ecology in action.

This example shows how the system handles a debugging scenario
where memories are stored, retrieved via activation spreading,
and consolidated over time.
"""

import asyncio

from src.core import Memory, MemoryType, MemoryLayer, WorkflowStep
from src.graph import MemoryGraph
from src.memory import EpisodicMemoryStore, ShortTermBuffer
from src.evolution import AntColonyWorkflowOptimizer
from src.evolution.strategies import EvolutionaryOptimizer
from src.swarm import SwarmCoordinator
from src.meta import MetaMemorySystem


async def demo_graph_memory():
    """Demonstrate activation-spreading retrieval."""
    print("=" * 60)
    print("DEMO: Activation-Based Graph Memory")
    print("=" * 60)

    graph = MemoryGraph()

    # Build a knowledge graph about a deployment issue
    memories = {
        "incident": Memory(id="incident", content="Production outage at 3am",
                          memory_type=MemoryType.EPISODIC),
        "cause": Memory(id="cause", content="Redis cluster ran out of memory",
                       memory_type=MemoryType.SEMANTIC),
        "fix": Memory(id="fix", content="Increased Redis maxmemory to 4GB",
                     memory_type=MemoryType.PROCEDURAL),
        "prevent": Memory(id="prevent", content="Set up memory alerting at 80%",
                         memory_type=MemoryType.PROCEDURAL),
        "similar": Memory(id="similar", content="Similar OOM issue last month with PostgreSQL",
                         memory_type=MemoryType.EPISODIC),
        "pattern": Memory(id="pattern", content="Memory issues often happen during batch jobs",
                         memory_type=MemoryType.SEMANTIC),
    }

    for mem in memories.values():
        graph.add_memory(mem)

    # Create associations
    graph.add_association("incident", "cause", weight=0.95, relation="caused_by")
    graph.add_association("cause", "fix", weight=0.9, relation="resolved_by")
    graph.add_association("fix", "prevent", weight=0.7, relation="leads_to")
    graph.add_association("cause", "similar", weight=0.6, relation="similar_to")
    graph.add_association("similar", "pattern", weight=0.8, relation="explained_by")

    # Query: Starting from the incident, what related memories activate?
    print("\nSeed: 'Production outage at 3am'")
    print("Spreading activation...\n")

    results = graph.get_activated_memories(seed_ids=["incident"], top_k=5)

    for mem, activation in results:
        bar = "█" * int(activation * 20)
        print(f"  [{activation:.2f}] {bar} {mem.content}")

    print()


def demo_ant_colony():
    """Demonstrate ant colony workflow optimization."""
    print("=" * 60)
    print("DEMO: Ant Colony Procedural Learning")
    print("=" * 60)

    colony = AntColonyWorkflowOptimizer()

    # Record successful debugging workflows
    for _ in range(5):
        colony.record_workflow(
            task="debug_error",
            steps=[
                WorkflowStep(tool="search", action="find_error_logs"),
                WorkflowStep(tool="analyze", action="trace_stacktrace"),
                WorkflowStep(tool="fix", action="apply_patch"),
                WorkflowStep(tool="test", action="run_tests"),
            ],
            success=True,
        )

    # Record a failed alternative
    colony.record_workflow(
        task="debug_error",
        steps=[
            WorkflowStep(tool="restart", action="restart_service"),
            WorkflowStep(tool="test", action="run_tests"),
        ],
        success=False,
    )

    # Now ask for suggestions
    print("\nPheromone trails after 5 successes + 1 failure:")
    print(f"  search → analyze: {colony.get_trail_strength('search', 'analyze'):.2f}")
    print(f"  analyze → fix:    {colony.get_trail_strength('analyze', 'fix'):.2f}")
    print(f"  fix → test:       {colony.get_trail_strength('fix', 'test'):.2f}")
    print(f"  restart → test:   {colony.get_trail_strength('restart', 'test'):.2f}")

    print(f"\nSuggested next step after 'search': "
          f"{colony.suggest_next_step('search', ['analyze', 'fix', 'restart', 'test'])}")
    print()


def demo_evolutionary_strategies():
    """Demonstrate evolutionary strategy optimization."""
    print("=" * 60)
    print("DEMO: Evolutionary Retrieval Strategies")
    print("=" * 60)

    optimizer = EvolutionaryOptimizer(population_size=8)

    # Simulate 5 generations of evolution
    for gen in range(5):
        # Evaluate each strategy
        for strategy in optimizer._population:
            # Simulate: semantic-heavy strategies perform better
            success = strategy.semantic_weight > 0.4
            strategy.evaluate(
                task_success=success,
                token_cost=50 + strategy.recency_weight * 100,
                hallucination=strategy.novelty_bonus > 0.4,
            )

        optimizer.evolve()
        stats = optimizer.population_stats
        print(f"  Gen {stats['generation']}: avg_fitness={stats['avg_fitness']:.3f}, "
              f"max_fitness={stats['max_fitness']:.3f}")

    best = optimizer.best_strategy
    if best:
        print(f"\nBest strategy: {best.name}")
        print(f"  recency={best.recency_weight:.2f}, semantic={best.semantic_weight:.2f}, "
              f"procedural={best.procedural_weight:.2f}")
    print()


def demo_meta_memory():
    """Demonstrate meta-memory self-assessment."""
    print("=" * 60)
    print("DEMO: Meta-Memory System")
    print("=" * 60)

    meta = MetaMemorySystem()

    # Register some sources
    meta.register_source("agent_episodic", "episodic_agent")
    meta.register_source("agent_semantic", "semantic_agent")
    meta.register_source("agent_procedural", "procedural_agent")

    # Simulate usage patterns
    for _ in range(10):
        meta.record_retrieval_quality("how to deploy", "episodic_agent",
                                      ["agent_episodic"], was_useful=True)
    for _ in range(8):
        meta.record_retrieval_quality("what is X", "semantic_agent",
                                      ["agent_semantic"], was_useful=True)
    for _ in range(3):
        meta.record_retrieval_quality("fix bug", "procedural_agent",
                                      ["agent_procedural"], was_useful=False,
                                      user_feedback="Outdated procedure")

    print("\nAgent Rankings:")
    for agent in ["episodic_agent", "semantic_agent", "procedural_agent"]:
        rank = meta.get_agent_ranking(agent)
        bar = "█" * int(rank * 20)
        print(f"  {agent:20s} [{rank:.2f}] {bar}")

    print("\nMeta Insights:")
    for insight in meta.generate_meta_insights():
        print(f"  • {insight}")

    unreliable = meta.get_unreliable_sources()
    if unreliable:
        print(f"\n⚠️  Unreliable sources: {[u.source_id for u in unreliable]}")
    print()


async def main():
    """Run all demos."""
    print("\n🧠 ADAPTIVE MEMORY ECOLOGY — PROTOTYPE DEMOS\n")

    await demo_graph_memory()
    demo_ant_colony()
    demo_evolutionary_strategies()
    demo_meta_memory()

    print("=" * 60)
    print("All demos complete. The system is ready for interactive use.")
    print("Run: python -m src.main")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
