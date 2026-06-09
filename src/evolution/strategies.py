"""Evolutionary strategy optimization for retrieval policies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalStrategy:
    """A single retrieval strategy organism in the evolutionary pool."""

    id: str
    name: str

    # Strategy weights (genes)
    recency_weight: float = 0.33
    semantic_weight: float = 0.33
    procedural_weight: float = 0.34
    novelty_bonus: float = 0.1
    activation_threshold: float = 0.2

    # Fitness tracking
    total_score: float = 0.0
    evaluations: int = 0
    generation: int = 0

    @property
    def fitness(self) -> float:
        if self.evaluations == 0:
            return 0.0
        return self.total_score / self.evaluations

    def evaluate(self, task_success: bool, token_cost: float, hallucination: bool) -> None:
        """Evaluate strategy on a task."""
        score = 0.0
        if task_success:
            score += 1.0
        score -= token_cost * 0.001  # Penalize token usage
        if hallucination:
            score -= 0.5
        self.total_score += max(0, score)
        self.evaluations += 1

    def mutate(self, mutation_rate: float = 0.1) -> RetrievalStrategy:
        """Create a mutated copy."""
        def mutate_val(v: float) -> float:
            delta = random.gauss(0, mutation_rate)
            return max(0.0, min(1.0, v + delta))

        return RetrievalStrategy(
            id=f"{self.id}_mut_{random.randint(0, 9999)}",
            name=f"{self.name} (mutant)",
            recency_weight=mutate_val(self.recency_weight),
            semantic_weight=mutate_val(self.semantic_weight),
            procedural_weight=mutate_val(self.procedural_weight),
            novelty_bonus=mutate_val(self.novelty_bonus),
            activation_threshold=mutate_val(self.activation_threshold),
            generation=self.generation + 1,
        )

    def crossover(self, other: RetrievalStrategy) -> RetrievalStrategy:
        """Create offspring from two parent strategies."""
        return RetrievalStrategy(
            id=f"cross_{random.randint(0, 9999)}",
            name=f"Crossover Gen {self.generation + 1}",
            recency_weight=random.choice([self.recency_weight, other.recency_weight]),
            semantic_weight=random.choice([self.semantic_weight, other.semantic_weight]),
            procedural_weight=random.choice([self.procedural_weight, other.procedural_weight]),
            novelty_bonus=(self.novelty_bonus + other.novelty_bonus) / 2,
            activation_threshold=random.choice(
                [self.activation_threshold, other.activation_threshold]
            ),
            generation=max(self.generation, other.generation) + 1,
        )


class EvolutionaryOptimizer:
    """Evolves retrieval strategies over time.

    Uses genetic algorithms to find optimal retrieval policies:
    - Population of strategies compete
    - Successful strategies reproduce
    - Failed strategies die off
    - Mutation introduces exploration
    """

    def __init__(self, population_size: int = 10, elite_count: int = 2):
        self._population_size = population_size
        self._elite_count = elite_count
        self._population: list[RetrievalStrategy] = self._initialize_population()
        self._generation = 0
        self._best_ever: RetrievalStrategy | None = None

    def _initialize_population(self) -> list[RetrievalStrategy]:
        """Create initial diverse population."""
        strategies = [
            RetrievalStrategy(id="recency_heavy", name="Recency Heavy",
                            recency_weight=0.7, semantic_weight=0.2, procedural_weight=0.1),
            RetrievalStrategy(id="semantic_heavy", name="Semantic Heavy",
                            recency_weight=0.1, semantic_weight=0.7, procedural_weight=0.2),
            RetrievalStrategy(id="procedural_heavy", name="Procedural Heavy",
                            recency_weight=0.1, semantic_weight=0.2, procedural_weight=0.7),
            RetrievalStrategy(id="balanced", name="Balanced",
                            recency_weight=0.33, semantic_weight=0.34, procedural_weight=0.33),
            RetrievalStrategy(id="novelty_seeker", name="Novelty Seeker",
                            recency_weight=0.2, semantic_weight=0.3, procedural_weight=0.2,
                            novelty_bonus=0.5),
        ]

        # Fill remaining with random strategies
        while len(strategies) < self._population_size:
            strategies.append(RetrievalStrategy(
                id=f"random_{len(strategies)}",
                name=f"Random {len(strategies)}",
                recency_weight=random.random(),
                semantic_weight=random.random(),
                procedural_weight=random.random(),
                novelty_bonus=random.random() * 0.3,
                activation_threshold=random.random() * 0.5,
            ))

        return strategies

    def get_active_strategy(self) -> RetrievalStrategy:
        """Get the current best strategy for use."""
        evaluated = [s for s in self._population if s.evaluations > 0]
        if not evaluated:
            return random.choice(self._population)
        return max(evaluated, key=lambda s: s.fitness)

    def evaluate_strategy(
        self,
        strategy_id: str,
        task_success: bool,
        token_cost: float = 0.0,
        hallucination: bool = False,
    ) -> None:
        """Record evaluation result for a strategy."""
        for strategy in self._population:
            if strategy.id == strategy_id:
                strategy.evaluate(task_success, token_cost, hallucination)
                break

    def evolve(self) -> None:
        """Run one generation of evolution."""
        self._generation += 1

        # Sort by fitness
        self._population.sort(key=lambda s: s.fitness, reverse=True)

        # Track best ever
        if self._population[0].fitness > 0:
            if self._best_ever is None or self._population[0].fitness > self._best_ever.fitness:
                self._best_ever = self._population[0]

        # Elite selection (keep top performers)
        new_population = self._population[:self._elite_count]

        # Fill rest through crossover and mutation
        while len(new_population) < self._population_size:
            if random.random() < 0.7:
                # Crossover
                parent_a = self._tournament_select()
                parent_b = self._tournament_select()
                child = parent_a.crossover(parent_b)
                if random.random() < 0.3:
                    child = child.mutate()
                new_population.append(child)
            else:
                # Mutation of random parent
                parent = random.choice(self._population[:5])
                new_population.append(parent.mutate())

        self._population = new_population

    def _tournament_select(self, tournament_size: int = 3) -> RetrievalStrategy:
        """Select parent via tournament selection."""
        candidates = random.sample(
            self._population, min(tournament_size, len(self._population))
        )
        return max(candidates, key=lambda s: s.fitness)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_strategy(self) -> RetrievalStrategy | None:
        return self._best_ever

    @property
    def population_stats(self) -> dict[str, Any]:
        fitnesses = [s.fitness for s in self._population if s.evaluations > 0]
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "max_fitness": max(fitnesses) if fitnesses else 0.0,
            "evaluated": len(fitnesses),
        }
