"""Memory store implementations - Episodic, Semantic, Procedural layers."""

from __future__ import annotations

import time
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core import Memory, MemoryLayer, MemoryType
from src.core.llm import get_embeddings
from config.settings import settings


class EpisodicMemoryStore:
    """Layer 1: Raw episodic memory with vector storage (ChromaDB)."""

    def __init__(self, collection_name: str = "episodic_memories"):
        self._client = chromadb.Client(
            ChromaSettings(
                persist_directory=settings.chroma_persist_dir,
                anonymized_telemetry=False,
            )
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = get_embeddings()
        self._memories: dict[str, Memory] = {}

    async def store(self, memory: Memory) -> str:
        """Store a memory with its embedding."""
        if memory.embedding is None:
            embedding = await self._embeddings.aembed_query(memory.content)
            memory.embedding = embedding

        self._collection.upsert(
            ids=[memory.id],
            embeddings=[memory.embedding],
            metadatas=[{
                "memory_type": memory.memory_type.value,
                "layer": memory.layer.value,
                "salience": memory.salience,
                "created_at": memory.created_at,
                "activation_level": memory.activation_level,
            }],
            documents=[memory.content],
        )
        self._memories[memory.id] = memory
        return memory.id

    async def retrieve(
        self,
        query: str,
        n_results: int = 10,
        min_activation: float = 0.0,
    ) -> list[Memory]:
        """Retrieve memories by semantic similarity."""
        query_embedding = await self._embeddings.aembed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"activation_level": {"$gte": min_activation}} if min_activation > 0 else None,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for mem_id in results["ids"][0]:
                if mem_id in self._memories:
                    memory = self._memories[mem_id]
                    memory.activate()
                    memories.append(memory)

        return memories

    def apply_decay(self) -> None:
        """Apply temporal decay to all memories."""
        for memory in self._memories.values():
            memory.decay(rate=settings.memory_decay_rate)

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    @property
    def size(self) -> int:
        return len(self._memories)


class ShortTermBuffer:
    """Fast temporary memory (hippocampus analog).

    Holds recent context with automatic overflow to episodic store.
    """

    def __init__(self, max_size: int = 50):
        self._buffer: list[Memory] = []
        self._max_size = max_size

    def add(self, memory: Memory) -> Memory | None:
        """Add memory to buffer. Returns overflow memory if buffer is full."""
        self._buffer.append(memory)
        overflow = None
        if len(self._buffer) > self._max_size:
            overflow = self._buffer.pop(0)
        return overflow

    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get most recent memories."""
        return self._buffer[-n:]

    def get_by_salience(self, n: int = 5) -> list[Memory]:
        """Get highest salience memories."""
        return sorted(self._buffer, key=lambda m: m.salience, reverse=True)[:n]

    def clear(self) -> list[Memory]:
        """Clear buffer and return all memories."""
        memories = self._buffer.copy()
        self._buffer.clear()
        return memories

    @property
    def size(self) -> int:
        return len(self._buffer)
