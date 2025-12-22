# Tubi Radar 🛰️

**Multi-Agent Swarm Competitive Intelligence System for Streaming Industry**

Tubi Radar is a local, multi-agent AI swarm that monitors competitors in the streaming/AVOD space using RSS feeds, parallel web search, specialist analysis agents, and OpenAI-powered synthesis.

---

## Features

- 🐝 **Swarm Architecture**: Parallel processing with 4 classifier workers, 5 search strategies
- 🎯 **Specialist Agents**: ThreatAnalyst, OpportunityFinder, TrendTracker, CompetitorProfiler
- 📡 **150+ Data Sources**: RSS feeds + Google News + parallel web search swarm
- 🔄 **Feedback Loops**: Critic agent reviews and validates analysis quality
- 🧠 **Vector Memory**: ChromaDB for context, competitor profiles, trend tracking
- 📊 **Executive Briefs**: Actionable intelligence reports for leadership
- 🔴 **Real-Time Streaming**: Continuous polling with breaking news alerts
- 🎨 **Tubi Dashboard**: Beautiful purple-themed UI at http://localhost:8000
- 🧪 **Test-First Framework**: Pytest suite with LLM mocking
- 🔧 **Fully Local**: No hosted orchestrators, runs entirely on your machine

---

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone the repository
cd /path/to/competitor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running Radar

```bash
# Full swarm mode (recommended) - parallel everything
python run_radar.py

# Quick mode (RSS only, fast)
python run_radar.py --quick

# Continuous streaming mode (runs forever)
python run_radar.py --stream

# Legacy smart pipeline mode
python run_radar.py --legacy

# Reset database before running
python run_radar.py --reset-db
```

### Running the Dashboard 🎨

```bash
# Start the Tubi-styled dashboard
python dashboard/app.py

# Or with uvicorn
python -m uvicorn dashboard.app:app --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## Swarm Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SWARM ORCHESTRATOR v2                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: Data Collection (Parallel)                                    │
│  ┌────────────────────┐  ┌────────────────────┐                         │
│  │   RSS Aggregator   │  │  Search Swarm      │                         │
│  │   (150+ feeds)     │  │  (5 strategies)    │                         │
│  │   - Competitors    │  │  - Breaking news   │                         │
│  │   - Google News    │  │  - Deals & M&A     │                         │
│  │   - Industry       │  │  - Product launches│                         │
│  └────────────────────┘  │  - Earnings        │                         │
│                          │  - Industry trends │                         │
│                          └────────────────────┘                         │
│                                    │                                     │
│  PHASE 2: Classification Swarm                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker 4 │                     │
│  │(50 arts)│  │(50 arts)│  │(50 arts)│  │(50 arts)│                     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                     │
│                                    │                                     │
│  PHASE 3: Specialist Analysis (Parallel)                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────┐               │
│  │ThreatAnalyst │  │OpportunityFinder │  │TrendTracker │               │
│  │  - Severity  │  │  - Value/Feasib  │  │ - Patterns  │               │
│  │  - Actions   │  │  - Action items  │  │ - Predict   │               │
│  └──────────────┘  └──────────────────┘  └─────────────┘               │
│           └─────────────┬─────────────────┘                             │
│                         │                                                │
│  PHASE 4: Critic Review ◄───────────────────┐                           │
│  ┌──────────────────────┐                   │                           │
│  │   Quality Control    │    Feedback Loop  │                           │
│  │   - Check evidence   │───────────────────┘                           │
│  │   - Validate claims  │                                               │
│  └──────────────────────┘                                               │
│                         │                                                │
│  PHASE 5: Memory Update                                                  │
│  ┌───────────────────────────────────────────┐                          │
│  │              ChromaDB                      │                          │
│  │  - Intel embeddings                        │                          │
│  │  - Competitor profiles                     │                          │
│  │  - Trend history                           │                          │
│  └───────────────────────────────────────────┘                          │
│                         │                                                │
│  PHASE 6: Synthesis                                                      │
│  ┌───────────────────────────────────────────┐                          │
│  │           Executive Brief                  │                          │
│  │  - TLDR                                    │                          │
│  │  - Critical Threats                        │                          │
│  │  - Strategic Opportunities                 │                          │
│  │  - Market Trends                           │                          │
│  │  - This Week's Priority                    │                          │
│  └───────────────────────────────────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Data Collection

| Component | Purpose | Output |
|-----------|---------|--------|
| **RSS Aggregator** | Parallel fetch of 150+ feeds | ArticleCandidate list |
| **Google News RSS** | Real-time news for each competitor | ArticleCandidate list |
| **Search Swarm** | 5 parallel search strategies | ArticleCandidate list |

### Classifier Swarm

4 parallel workers using gpt-4o-mini for speed:
- Each processes ~50 articles simultaneously
- Results merged and deduplicated
- Categories: strategic, product, content, marketing, ai_ads, pricing

### Specialist Agents

| Agent | Role | Output |
|-------|------|--------|
| **ThreatAnalyst** | Identifies competitive threats | ThreatAssessment (severity, actions) |
| **OpportunityFinder** | Finds gaps and opportunities | Opportunity (value, feasibility) |
| **TrendTracker** | Pattern recognition, predictions | Trend (direction, strength) |
| **CompetitorProfiler** | Live competitor profiles | CompetitorProfile (strategy, threat level) |

### Critic Agent

- Reviews analyst output for quality
- Checks for missing context
- Validates claims
- Approves or requests revision

### Vector Memory

ChromaDB collections:
- `intel_embeddings`: All intel with semantic search
- `competitor_profiles`: Strategy tracking over time
- `trends_history`: Trend evolution tracking

---

## Configuration

### `config/radar.yaml`

```yaml
competitors:
  - id: netflix
    name: "Netflix"
    category: streaming
    feeds:
      - label: netflix_newsroom
        type: rss
        url: "https://about.netflix.com/en/newsroom/rss"
      - label: netflix_google_news
        type: rss
        url: "https://news.google.com/rss/search?q=Netflix+streaming+news"
    search_queries:
      - "Netflix ad-supported tier"
      - "Netflix originals announcements"

industry_feeds:
  - label: streaming_wars_news
    type: rss
    url: "https://news.google.com/rss/search?q=streaming+wars+2025"
  - label: variety
    type: rss
    url: "https://variety.com/feed/"
    filter_keywords:
      - streaming
      - FAST
      - AVOD
      - Tubi

global:
  lookback_hours: 48
  max_articles_per_feed: 20
  max_web_searches: 30
  min_relevance_score: 3.0
  min_impact_score: 3.0
  
  models:
    reasoning: "gpt-4o"
    structured: "gpt-4o-mini"
    embedding: "text-embedding-3-small"
```

---

## CLI Options

```bash
python run_radar.py [OPTIONS]

Options:
  (none)        Full swarm mode (default, recommended)
  --quick       Quick mode (RSS only, no specialists)
  --stream      Continuous streaming mode
  --legacy      Legacy smart pipeline mode
  --reset-db    Reset database before running
```

---

## Performance

| Metric | Current |
|--------|---------|
| Articles collected | 700-1000+ |
| Intel items | 150-200+ |
| Threats identified | 5-15 |
| Opportunities found | 3-10 |
| Trends tracked | 5-10 |
| Processing time | 2-3 minutes |
| Source coverage | 150+ sources |

---

## Output

### Report Structure

```markdown
# Executive Brief for Tubi Leadership

## TLDR
[2-3 sentence summary of most important insight]

## Critical Threats
1. **[Threat Name]**
   - Severity: X/10
   - Impact: [Description]
   - Recommended Action: [Action]

## Strategic Opportunities
1. **[Opportunity Name]**
   - Value: X/10
   - Feasibility: X/10
   - Next Steps: [Actions]

## Market Trends
- [Trend 1]: [Description]
- [Trend 2]: [Description]

## This Week's Priority
[Single most important focus]
```

---

## Streaming Mode

Run continuous intelligence gathering:

```bash
python run_radar.py --stream
```

Features:
- RSS polling every 15 minutes
- Web search sweep every hour
- Breaking news detection
- High-impact alerts (impact >= 7)
- Live dashboard updates

---

## Monitored Sources (150+)

### Streaming Services (20+)
YouTube, Netflix, Disney+/Hulu, Prime Video/Freevee, Roku, Paramount+/Pluto TV, Peacock, Max/HBO, Apple TV+, Plex, Fubo TV, Sling TV, DirecTV, ESPN+, AMC+, Xumo, ViX, BritBox, Crunchyroll

### Google News (Real-time)
- Competitor-specific feeds for all major players
- Industry-wide: streaming wars, FAST channels, AVOD, CTV advertising, cord cutting

### Industry News (25+)
Variety, Deadline, Hollywood Reporter, The Verge, TechCrunch, Engadget, Ars Technica, WIRED, CNET, Cord Cutters News, The Streamable, NextTV, Fierce Video, AdWeek, Digiday, AdExchanger

---

## Project Structure

```
competitor/
├── config/
│   └── radar.yaml           # Main configuration
├── data/
│   ├── radar.db             # SQLite database
│   └── chroma/              # Vector store
├── dashboard/
│   ├── app.py               # Flask app
│   ├── static/css/style.css # Tubi-themed styles
│   └── templates/           # HTML templates
├── radar/
│   ├── config.py            # Config loading
│   ├── database.py          # Database management
│   ├── models.py            # SQLAlchemy models
│   ├── stream.py            # Real-time streaming engine
│   ├── agents/
│   │   ├── classifier_swarm.py  # 4 parallel classifiers
│   │   ├── search_swarm.py      # 5 search strategies
│   │   ├── critic.py            # Quality feedback loops
│   │   ├── orchestrator_v2.py   # Swarm orchestrator
│   │   └── specialists/
│   │       ├── threat.py        # Threat analysis
│   │       ├── opportunity.py   # Opportunity finder
│   │       ├── trends.py        # Trend tracking
│   │       └── profiler.py      # Competitor profiles
│   └── tools/
│       ├── rss.py           # RSS fetching
│       ├── web_search.py    # OpenAI web search
│       ├── db_tools.py      # Database tools
│       └── vector.py        # ChromaDB + memory
├── reports/                  # Generated reports
├── tests/                    # Test suite
├── run_radar.py             # CLI entry point
├── requirements.txt
└── README.md
```

---

## Testing

```bash
# Run unit tests (no API calls)
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=radar

# Run integration tests (requires API key)
pytest tests/integration/ --run-integration
```

---

## Requirements

```
langchain>=0.3.0
langchain-openai>=0.3.0
langgraph>=0.2.0
openai>=1.0.0
chromadb>=0.5.0
sqlalchemy>=2.0.0
feedparser>=6.0.0
httpx>=0.27.0
pyyaml>=6.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
beautifulsoup4>=4.12.0
flask>=3.0.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

---

## License

Internal use only - Tubi/Fox Corporation
