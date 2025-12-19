# Tubi Radar

**Multi-Agent Competitive Intelligence System**

A local-only, multi-agent system for monitoring competitive intelligence in the streaming industry. Built with LangGraph, ChromaDB, and OpenAI.

## Features

- **Multi-Agent Architecture**: Specialized agents for ingestion, classification, memory, domain analysis, and report generation
- **Blackboard Pattern**: Shared state via SQLite + ChromaDB for agent coordination
- **LangGraph Orchestration**: DAG-based workflow management
- **Structured Outputs**: Reliable JSON schemas for LLM responses
- **Local-Only**: No external services beyond OpenAI API

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Run Radar

```bash
# Initialize database and run Phase 1 pipeline
python run_radar.py

# Or run the full multi-agent pipeline
python run_radar.py --full
```

## Project Structure

```
sro/
├── config/
│   └── radar.yaml              # Competitor feeds & settings
├── data/
│   ├── radar.db                # SQLite database
│   └── chroma/                 # ChromaDB persistence
├── reports/                    # Generated markdown reports
├── radar/
│   ├── config.py               # YAML config loader
│   ├── models.py               # SQLAlchemy models
│   ├── database.py             # DB session management
│   ├── schemas.py              # Pydantic schemas for LLM outputs
│   ├── graph.py                # LangGraph workflow
│   ├── tools/
│   │   ├── rss.py              # RSS fetching
│   │   ├── html.py             # HTML excerpt extraction
│   │   ├── db_tools.py         # DB read/write tools
│   │   └── vector.py           # ChromaDB operations
│   └── agents/
│       ├── base.py             # Base agent class
│       ├── ingestion.py        # Collector agent
│       ├── understanding.py    # Classifier agent
│       ├── memory.py           # Archivist agent
│       ├── domain.py           # Domain-specific agents
│       └── editor.py           # Report synthesizer
└── run_radar.py                # CLI entrypoint
```

## Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Ingestion** | Collector | Fetches articles from RSS feeds and web search |
| **Understanding** | Classifier | Summarizes, categorizes, and scores articles |
| **Memory** | Archivist | Deduplicates and computes novelty scores |
| **Product** | Domain Expert | Analyzes product/UX implications |
| **Content** | Domain Expert | Analyzes content strategy implications |
| **Marketing** | Domain Expert | Analyzes marketing/positioning implications |
| **AI/Ads** | Domain Expert | Analyzes ad tech/pricing implications |
| **Editor** | Synthesizer | Generates executive report |

## Configuration

Edit `config/radar.yaml` to customize:

- **Competitors**: RSS feeds and search queries per competitor
- **Industry Feeds**: General streaming news sources
- **Thresholds**: Relevance, impact, and novelty score minimums
- **Models**: OpenAI model selection for different tasks

## CLI Options

```bash
python run_radar.py [OPTIONS]

Options:
  --phase1          Run Phase 1 pipeline only (default)
  --full            Run full multi-agent pipeline
  --web-search      Enable web search during ingestion
  --no-memory       Disable memory/deduplication phase
  --no-domain       Disable domain agent analysis
  --init-db         Initialize database only
  -v, --verbose     Enable verbose output
```

## Output

Reports are generated in `reports/` as Markdown files:

```
reports/radar-2025-12-19-run1.md
```

Each report includes:
- Top 3-5 key competitive moves
- Product & UX section
- Content & Library section
- Marketing & Positioning section
- AI & Ads / Pricing section
- Suggested actions

## Database Schema

| Table | Description |
|-------|-------------|
| `runs` | Track each radar execution |
| `articles` | Raw source items from feeds |
| `intel` | Classified and scored items |
| `annotations` | Domain agent commentary |
| `reports` | Generated report metadata |

## Requirements

- Python 3.11+
- OpenAI API key
- ~100MB disk space for ChromaDB

## License

Internal use only.

