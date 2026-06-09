"""Unit tests for core memory models."""

import time

import pytest

from src.core import Memory, MemoryLayer, MemoryType, Workflow, WorkflowStep


class TestMemory:
    def test_create_memory(self):
        mem = Memory(content="test memory", memory_type=MemoryType.EPISODIC)
        assert mem.content == "test memory"
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.layer == MemoryLayer.RAW_EPISODIC
        assert mem.activation_level == 1.0
        assert mem.access_count == 0

    def test_memory_decay(self):
        mem = Memory(content="test", memory_type=MemoryType.EPISODIC)
        mem.last_accessed = time.time() - 7200  # 2 hours ago
        mem.decay(rate=0.1)
        assert mem.activation_level < 1.0

    def test_memory_activate(self):
        mem = Memory(content="test", memory_type=MemoryType.EPISODIC)
        mem.activation_level = 0.5
        mem.activate(boost=0.3)
        assert mem.activation_level == 0.8
        assert mem.access_count == 1

    def test_activation_capped_at_1(self):
        mem = Memory(content="test", memory_type=MemoryType.EPISODIC)
        mem.activate(boost=0.5)
        assert mem.activation_level == 1.0


class TestWorkflow:
    def test_create_workflow(self):
        steps = [
            WorkflowStep(tool="search", action="find"),
            WorkflowStep(tool="edit", action="modify"),
        ]
        wf = Workflow(steps=steps)
        assert len(wf.steps) == 2
        assert wf.pheromone_strength == 1.0

    def test_reinforce(self):
        wf = Workflow(steps=[WorkflowStep(tool="a", action="b")])
        wf.reinforce(amount=0.5)
        assert wf.pheromone_strength == 1.5
        assert wf.success_count == 1

    def test_weaken(self):
        wf = Workflow(steps=[WorkflowStep(tool="a", action="b")])
        wf.weaken(amount=0.3)
        assert wf.pheromone_strength == 0.7
        assert wf.failure_count == 1

    def test_success_rate(self):
        wf = Workflow(steps=[WorkflowStep(tool="a", action="b")])
        wf.reinforce()
        wf.reinforce()
        wf.weaken()
        assert wf.success_rate == pytest.approx(2 / 3)
