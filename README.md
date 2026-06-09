# Adaptive Memory Ecology

> Think of memory as an ecosystem — not a database.

A biologically-inspired, multi-layer memory architecture for autonomous AI agents. Goes beyond standard RAG by implementing activation spreading, competitive retrieval, evolutionary optimization, swarm processing, and memory consolidation cycles.

---

## Table of Contents

- [Architecture](#architecture)
- [What's Implemented](#whats-implemented)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Components Deep Dive](#components-deep-dive)
- [Testing](#testing)
- [Research References](#research-references)
- [License](#license)

---

## Architecture

```
┌─────────────────┐
│ Executive Agent  │
└────────┬────────┘
         ↓
┌─────────────────────────┐
│ Meta Retrieval Planner  │
└────────┬────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ Distributed Specialized Memory Agents   │
└─────────────────────────────────────────┘
     ↓         ↓           ↓          ↓
 Episodic   Semantic   Procedural   Social
 Memory     Memory     Memory       Memory
     ↓         ↓           ↓          ↓
┌─────────────────────────────────────────┐
│   Adaptive Retrieval Competition Layer  │
└─────────────────────────────────────────┘
         ↓
   Swarm Processing
         ↓
   Memory Consolidation
         ↓
   Reinforcement Loop
```

---

## What's Implemented

| Component | Status | Description |
|-----------|--------|-------------|
| **Hierarchical Memory Store** | ✅ | 5-layer memory (episodic → summary → semantic → procedural → meta) |
| **Short-Term Buffer** | ✅ | Hippocampus-analog with overflow to long-term store |
| **Vector Store (ChromaDB)** | ✅ | Semantic similarity retrieval with embeddings |
| **Graph Memory (NetworkX)** | ✅ | Activation-spreading retrieval across associations |
| **Specialized Agents** | ✅ | Episodic, Semantic, Procedural, Social retrieval agents |
| **Competitive Retrieval** | ✅ | Immune-system-inspired agent bidding and arbitration |
| **Ant Colony Optimization** | ✅ | Pheromone-based workflow learning and suggestion |
| **Evolutionary Strategies** | ✅ | Genetic algorithm optimization of retrieval policies |
| **Swarm Processing** | ✅ | Parallel agents for salience, contradiction, decay, cleaning |
| **Consolidation Engine** | ✅ | Sleep cycles with compression, concept extraction, pruning |
| **Meta-Memory** | ✅ | Self-assessment of memory quality and agent reliability |
| **Memory Homeostasis** | ✅ | Temporal decay, aggressive forgetting, salience reinforcement |
| **LangGraph Orchestration** | ✅ | Full state-machine pipeline (ingest → retrieve → swarm → generate → feedback) |
| **Interactive CLI** | ✅ | Rich terminal interface for querying and managing the system |
| **Configurable LLM** | ✅ | Supports both OpenAI and Anthropic |

---

## Prerequisites

- **Python 3.11+** (tested with 3.13)
- **Redis** (optional — for short-term memory persistence; system works without it)
- **OpenAI API key** and/or **Anthropic API key** (for LLM-powered features)

### System Dependencies

```bash
# macOS
brew install python@3.13 redis

# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv redis-server

# Windows
# Install Python 3.11+ from python.org
# Install Redis via WSL or Memurai
```

---

## Installation

### 1. Clone and navigate to the project

```bash
cd "Agentic Memory"
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install the project and all dependencies

```bash
pip install -e ".[dev]"
```

This installs all required packages:

#### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | ≥0.2.0 | State-machine orchestration |
| `langchain` | ≥0.3.0 | LLM framework and chains |
| `langchain-openai` | ≥0.2.0 | OpenAI integration |
| `langchain-anthropic` | ≥0.2.0 | Anthropic/Claude integration |
| `langchain-community` | ≥0.3.0 | Community integrations |
| `langmem` | ≥0.0.14 | Memory extraction utilities |
| `chromadb` | ≥0.5.0 | Vector database (embedded) |
| `networkx` | ≥3.3 | Graph memory and activation spreading |
| `redis` | ≥5.0 | Short-term memory store |
| `numpy` | ≥1.26 | Numerical operations |
| `pydantic` | ≥2.7 | Data models and validation |
| `pydantic-settings` | ≥2.3 | Configuration management |
| `python-dotenv` | ≥1.0 | Environment variable loading |
| `tiktoken` | ≥0.7 | Token counting |
| `tenacity` | ≥8.3 | Retry logic |
| `structlog` | ≥24.1 | Structured logging |
| `rich` | ≥13.7 | Terminal UI and formatting |

#### Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥8.2 | Testing framework |
| `pytest-asyncio` | ≥0.23 | Async test support |
| `pytest-cov` | ≥5.0 | Coverage reporting |
| `ruff` | ≥0.4 | Linting and formatting |
| `mypy` | ≥1.10 | Static type checking |

---

## Configuration

### 1. Create your environment file

```bash
cp .env.example .env
```

### 2. Edit `.env` with your API keys

```env
# LLM Providers (at least one required for full functionality)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Default LLM Provider: "openai" or "anthropic"
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# Redis (optional — start Redis locally or skip for in-memory only)
REDIS_URL=redis://localhost:6379/0

# ChromaDB persistence directory
CHROMA_PERSIST_DIR=./data/chroma

# Memory Tuning
MEMORY_DECAY_RATE=0.05
CONSOLIDATION_INTERVAL_SECONDS=3600
MAX_EPISODIC_MEMORIES=10000
ACTIVATION_THRESHOLD=0.3

# Logging
LOG_LEVEL=INFO
```

### 3. (Optional) Start Redis

```bash
redis-server --daemonize yes
```

---

## Running the Project

### Run the Demo (no API keys needed)

Demonstrates graph memory, ant colony optimization, evolutionary strategies, and meta-memory — all running locally without LLM calls:

```bash
python examples/demo.py
```

**Expected output:**
```
🧠 ADAPTIVE MEMORY ECOLOGY — PROTOTYPE DEMOS

DEMO: Activation-Based Graph Memory
  [0.57] ███████████ Redis cluster ran out of memory
  [0.31] ██████ Increased Redis maxmemory to 4GB
  [0.21] ████ Similar OOM issue last month with PostgreSQL

DEMO: Ant Colony Procedural Learning
  search → analyze: 1.51
  restart → test:   0.01
  Suggested next step after 'search': analyze

DEMO: Evolutionary Retrieval Strategies
  Gen 5: avg_fitness=0.950, max_fitness=0.950
  Best strategy: semantic=0.92

DEMO: Meta-Memory System
  Overall retrieval success rate: 85.7%
  ⚠️  Unreliable sources: ['agent_procedural']
```

### Run Interactive Mode (requires API keys)

Full LLM-powered interactive session with all memory ecology features:

```bash
python -m src.main
```

**Available commands:**
| Command | Description |
|---------|-------------|
| `query <text>` | Query the memory system with full retrieval pipeline |
| `add <text>` | Store a new memory in the ecology |
| `consolidate` | Run a sleep/consolidation cycle |
| `health` | Display system health metrics |
| `feedback y/n` | Provide feedback on the last response |
| `quit` | Exit |

### Run as a Module

```bash
# Import and use programmatically
python -c "
import asyncio
from src.core.orchestrator import MemoryEcology

async def main():
    ecology = MemoryEcology()
    await ecology.add_memory('Redis OOM caused production outage')
    await ecology.add_memory('Fixed by increasing maxmemory to 4GB')
    response = await ecology.query('What caused the outage?')
    print(response)

asyncio.run(main())
"
```

---

## Project Structure

```
.
├── README.md                       # This file
├── pyproject.toml                  # Project metadata and dependencies
├── .env.example                    # Environment template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # Interactive CLI entry point
│   │
│   ├── core/
│   │   ├── __init__.py             # Data models (Memory, Workflow, AgentState)
│   │   ├── llm.py                  # LLM provider abstraction (OpenAI/Anthropic)
│   │   └── orchestrator.py         # LangGraph state machine orchestrator
│   │
│   ├── memory/
│   │   └── __init__.py             # EpisodicMemoryStore + ShortTermBuffer
│   │
│   ├── graph/
│   │   └── __init__.py             # MemoryGraph with activation spreading
│   │
│   ├── agents/
│   │   └── __init__.py             # Specialized retrieval agents
│   │
│   ├── retrieval/
│   │   └── __init__.py             # RetrievalArbitrator (competitive retrieval)
│   │
│   ├── evolution/
│   │   ├── __init__.py             # AntColonyWorkflowOptimizer
│   │   └── strategies.py           # Evolutionary retrieval strategies
│   │
│   ├── consolidation/
│   │   └── __init__.py             # ConsolidationEngine (sleep cycles)
│   │
│   ├── swarm/
│   │   └── __init__.py             # SwarmCoordinator + specialized swarm agents
│   │
│   ├── meta/
│   │   └── __init__.py             # MetaMemorySystem (Layer 5)
│   │
│   └── utils/
│       └── __init__.py
│
├── tests/
│   ├── unit/
│   │   ├── test_core.py            # Memory model and workflow tests
│   │   ├── test_graph.py           # Graph activation spreading tests
│   │   └── test_evolution.py       # Evolutionary optimizer tests
│   └── integration/
│
├── examples/
│   └── demo.py                     # Full demo showcasing all components
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Pydantic settings from .env
│
└── docs/
```

---

## Components Deep Dive

### Hierarchical Memory Layers

| Layer | Name | What It Stores | How It's Created |
|-------|------|----------------|------------------|
| 1 | Raw Episodic | Exact conversations/events | Direct ingestion |
| 2 | Summaries | Compressed episode clusters | Consolidation engine |
| 3 | Semantic Concepts | Extracted knowledge/principles | LLM concept extraction |
| 4 | Procedural Policies | Behavioral rules/workflows | Ant colony reinforcement |
| 5 | Meta-Memory | Knowledge about memory quality | Self-assessment loop |

### Retrieval Pipeline

```
Query → Ingest to Short-Term Buffer
     → Agents Bid on Relevance (immune-system competition)
     → Top Agents Retrieve in Parallel
     → Swarm Processing (clean, validate, score)
     → LLM Generates Response
     → Feedback Updates Agent Rankings
```

### Memory Homeostasis

The system implements biological forgetting:
- **Temporal decay** — unused memories lose activation over time
- **Pruning** — memories below threshold are permanently removed
- **Consolidation** — related memories compress into abstractions
- **Reinforcement** — accessed memories get activation boosts

---

## Testing

### Run all tests

```bash
pytest
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=src --cov-report=term-missing
```

### Run linter

```bash
ruff check src/ tests/
```

### Run type checker

```bash
mypy src/
```

---

## Research References

### Cognitive Architectures
- ACT-R
- SOAR
- Global Workspace Theory
- Predictive Processing

### Bio-Inspired Computing
- Ant Colony Optimization
- Hopfield Networks
- Neural Turing Machines
- Hierarchical Temporal Memory
- Sparse Distributed Memory

### Memory/Retrieval Research
- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- Hippocampal Memory Indexing Theory

### Recommended Learning
- DeepLearning.AI — Agentic AI Courses
- Harrison Chase — Long-Term Agentic Memory with LangGraph
- Joscha Bach — Cognitive Architectures

---

## Key Design Insight

> **Do NOT build:** Better vector search  
> **Build:** Adaptive memory ecology  

The winning systems will combine: graph memory, memory consolidation, procedural evolution, swarm retrieval, meta-memory, reinforcement loops, activation spreading, and hierarchical abstraction. **Not just embeddings.**

---

## Contributing

Contributions welcome. This is an active research project exploring biologically-plausible agentic cognition.
