"""Ant Colony Optimization for procedural memory and workflow learning."""

from __future__ import annotations

import random
from typing import Any

from src.core import Workflow, WorkflowStep


class AntColonyWorkflowOptimizer:
    """Learns optimal workflows through pheromone reinforcement.

    Successful workflows get strengthened, failed ones decay.
    Over time, optimal action sequences emerge.
    """

    def __init__(
        self,
        evaporation_rate: float = 0.1,
        min_pheromone: float = 0.01,
        max_pheromone: float = 10.0,
    ):
        self._workflows: dict[str, Workflow] = {}
        self._evaporation_rate = evaporation_rate
        self._min_pheromone = min_pheromone
        self._max_pheromone = max_pheromone

        # Pheromone trails: maps (tool_from, tool_to) -> strength
        self._pheromone_trails: dict[tuple[str, str], float] = {}

        # Task -> workflow mapping
        self._task_workflows: dict[str, list[str]] = {}

    def record_workflow(self, task: str, steps: list[WorkflowStep], success: bool) -> Workflow:
        """Record a completed workflow and update pheromones."""
        workflow = Workflow(steps=steps)

        if success:
            workflow.reinforce()
            self._reinforce_trail(steps)
        else:
            workflow.weaken()
            self._weaken_trail(steps)

        self._workflows[workflow.id] = workflow

        if task not in self._task_workflows:
            self._task_workflows[task] = []
        self._task_workflows[task].append(workflow.id)

        return workflow

    def get_best_workflows(self, task: str, top_k: int = 3) -> list[tuple[Workflow, float]]:
        """Get best workflows for a task type."""
        workflow_ids = self._task_workflows.get(task, [])
        workflows = [
            (self._workflows[wid], self._workflows[wid].pheromone_strength)
            for wid in workflow_ids
            if wid in self._workflows
        ]
        workflows.sort(key=lambda x: x[1], reverse=True)
        return workflows[:top_k]

    def suggest_next_step(self, current_tool: str, available_tools: list[str]) -> str:
        """Suggest the next tool based on pheromone trails."""
        if not available_tools:
            return current_tool

        # Get pheromone strengths for each possible next step
        weights = []
        for tool in available_tools:
            key = (current_tool, tool)
            pheromone = self._pheromone_trails.get(key, self._min_pheromone)
            weights.append(pheromone)

        # Probabilistic selection weighted by pheromone
        total = sum(weights)
        if total == 0:
            return random.choice(available_tools)

        probabilities = [w / total for w in weights]
        return random.choices(available_tools, weights=probabilities, k=1)[0]

    def evaporate(self) -> None:
        """Apply global pheromone evaporation (forgetting)."""
        for key in list(self._pheromone_trails.keys()):
            self._pheromone_trails[key] *= (1 - self._evaporation_rate)
            if self._pheromone_trails[key] < self._min_pheromone:
                self._pheromone_trails[key] = self._min_pheromone

    def _reinforce_trail(self, steps: list[WorkflowStep], amount: float = 0.3) -> None:
        """Reinforce pheromone trail for successful workflow."""
        for i in range(len(steps) - 1):
            key = (steps[i].tool, steps[i + 1].tool)
            current = self._pheromone_trails.get(key, self._min_pheromone)
            self._pheromone_trails[key] = min(self._max_pheromone, current + amount)

    def _weaken_trail(self, steps: list[WorkflowStep], amount: float = 0.1) -> None:
        """Weaken pheromone trail for failed workflow."""
        for i in range(len(steps) - 1):
            key = (steps[i].tool, steps[i + 1].tool)
            current = self._pheromone_trails.get(key, self._min_pheromone)
            self._pheromone_trails[key] = max(self._min_pheromone, current - amount)

    def get_trail_strength(self, from_tool: str, to_tool: str) -> float:
        """Get current pheromone strength between two tools."""
        return self._pheromone_trails.get((from_tool, to_tool), self._min_pheromone)

    @property
    def workflow_count(self) -> int:
        return len(self._workflows)

    @property
    def trail_count(self) -> int:
        return len(self._pheromone_trails)
