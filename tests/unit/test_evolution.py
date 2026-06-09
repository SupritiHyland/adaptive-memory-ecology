"""Unit tests for evolutionary strategy optimization."""

import pytest

from src.evolution.strategies import EvolutionaryOptimizer, RetrievalStrategy


class TestRetrievalStrategy:
    def test_evaluate(self):
        strategy = RetrievalStrategy(id="test", name="Test")
        strategy.evaluate(task_success=True, token_cost=100, hallucination=False)
        assert strategy.evaluations == 1
        assert strategy.fitness > 0

    def test_mutate(self):
        strategy = RetrievalStrategy(id="parent", name="Parent",
                                     recency_weight=0.5, semantic_weight=0.3)
        mutant = strategy.mutate(mutation_rate=0.1)
        assert mutant.id != strategy.id
        assert mutant.generation == strategy.generation + 1

    def test_crossover(self):
        a = RetrievalStrategy(id="a", name="A", recency_weight=0.9, semantic_weight=0.1)
        b = RetrievalStrategy(id="b", name="B", recency_weight=0.1, semantic_weight=0.9)
        child = a.crossover(b)
        assert child.generation == 1
        # Child's weights should come from either parent
        assert child.recency_weight in (0.9, 0.1)


class TestEvolutionaryOptimizer:
    def test_initialization(self):
        optimizer = EvolutionaryOptimizer(population_size=10)
        assert optimizer.generation == 0
        assert len(optimizer._population) == 10

    def test_evolve(self):
        optimizer = EvolutionaryOptimizer(population_size=5)
        # Evaluate some strategies
        for s in optimizer._population:
            s.evaluate(task_success=True, token_cost=50, hallucination=False)
        optimizer.evolve()
        assert optimizer.generation == 1

    def test_get_active_strategy(self):
        optimizer = EvolutionaryOptimizer(population_size=5)
        strategy = optimizer.get_active_strategy()
        assert strategy is not None
        assert isinstance(strategy, RetrievalStrategy)

    def test_population_stats(self):
        optimizer = EvolutionaryOptimizer(population_size=5)
        stats = optimizer.population_stats
        assert stats["population_size"] == 5
        assert stats["generation"] == 0
