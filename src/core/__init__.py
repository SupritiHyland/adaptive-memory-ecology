"""Core data models for the memory ecology system."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SOCIAL = "social"
    META = "meta"


class MemoryLayer(int, Enum):
    RAW_EPISODIC = 1
    SUMMARY = 2
    SEMANTIC_CONCEPT = 3
    PROCEDURAL_POLICY = 4
    META_MEMORY = 5


class Memory(BaseModel):
    """A single memory unit in the ecology."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType
    layer: MemoryLayer = MemoryLayer.RAW_EPISODIC
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    # Activation and decay
    activation_level: float = 1.0
    salience: float = 0.5
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)
    created_at: float = Field(default_factory=time.time)

    # Graph connections
    related_memory_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)

    def decay(self, rate: float = 0.05) -> None:
        """Apply temporal decay to activation level."""
        elapsed = time.time() - self.last_accessed
        self.activation_level *= (1 - rate) ** (elapsed / 3600)
        self.activation_level = max(0.01, self.activation_level)

    def activate(self, boost: float = 0.2) -> None:
        """Boost activation on access."""
        self.access_count += 1
        self.last_accessed = time.time()
        self.activation_level = min(1.0, self.activation_level + boost)


class WorkflowStep(BaseModel):
    """A single step in a procedural workflow."""

    tool: str
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    success: bool = True


class Workflow(BaseModel):
    """A recorded workflow with pheromone strength."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[WorkflowStep]
    pheromone_strength: float = 1.0
    success_count: int = 0
    failure_count: int = 0
    created_at: float = Field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def reinforce(self, amount: float = 0.1) -> None:
        """Strengthen pheromone on success."""
        self.success_count += 1
        self.pheromone_strength = min(10.0, self.pheromone_strength + amount)

    def weaken(self, amount: float = 0.2) -> None:
        """Weaken pheromone on failure."""
        self.failure_count += 1
        self.pheromone_strength = max(0.01, self.pheromone_strength - amount)


class RetrievalResult(BaseModel):
    """Result from a retrieval agent."""

    memory: Memory
    relevance_score: float
    confidence: float
    novelty: float = 0.5
    agent_id: str = ""


class AgentState(BaseModel):
    """State passed through the LangGraph execution."""

    query: str = ""
    context: list[Memory] = Field(default_factory=list)
    retrieved_memories: list[RetrievalResult] = Field(default_factory=list)
    response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_step: str = "init"
