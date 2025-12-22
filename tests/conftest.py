"""
Pytest configuration and shared fixtures for Tubi Radar tests.

Provides:
- Database fixtures (in-memory SQLite)
- Config fixtures (test configuration)
- LLM mocking fixtures
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config_dict() -> dict:
    """Return a minimal test configuration dictionary."""
    return {
        "competitors": [
            {
                "id": "test_competitor",
                "name": "Test Competitor",
                "category": "streaming",
                "feeds": [
                    {
                        "label": "test_feed",
                        "type": "rss",
                        "url": "https://example.com/feed.xml",
                    }
                ],
                "search_queries": ["test query"],
            }
        ],
        "industry_feeds": [
            {
                "label": "test_industry",
                "type": "rss",
                "url": "https://example.com/industry.xml",
                "filter_keywords": ["streaming", "video"],
            }
        ],
        "global": {
            "lookback_hours": 24,
            "max_articles_per_feed": 10,
            "feed_timeout": 5,
            "max_concurrent_feeds": 2,
            "min_relevance_score": 4.0,
            "min_impact_score": 4.0,
            "max_items_total": 50,
            "max_report_items": 10,
            "enable_web_search": False,
            "models": {
                "reasoning": "gpt-4o",
                "structured": "gpt-4o",
                "embedding": "text-embedding-3-small",
            },
        },
    }


@pytest.fixture
def test_config_file(test_config_dict, tmp_path) -> Path:
    """Create a temporary config file for testing."""
    import yaml
    
    config_path = tmp_path / "radar.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config_dict, f)
    
    return config_path


@pytest.fixture
def mock_config(test_config_dict):
    """Mock the config module to return test configuration."""
    from radar.config import RadarConfig
    
    config = RadarConfig.model_validate(test_config_dict)
    
    with patch("radar.config._config", config):
        with patch("radar.config.get_config", return_value=config):
            yield config


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def test_db(tmp_path) -> Generator[Path, None, None]:
    """Create a temporary test database."""
    db_path = tmp_path / "test_radar.db"
    
    # Set environment variable for database path
    old_path = os.environ.get("RADAR_DB_PATH")
    os.environ["RADAR_DB_PATH"] = str(db_path)
    
    # Reset the global engine/session factory
    import radar.database as db_module
    db_module._engine = None
    db_module._SessionFactory = None
    
    # Initialize the database
    from radar.database import init_database
    init_database(db_path)
    
    yield db_path
    
    # Cleanup
    if old_path:
        os.environ["RADAR_DB_PATH"] = old_path
    else:
        os.environ.pop("RADAR_DB_PATH", None)
    
    db_module._engine = None
    db_module._SessionFactory = None


@pytest.fixture
def db_session(test_db):
    """Get a database session for testing."""
    from radar.database import get_session
    
    with get_session() as session:
        yield session


# =============================================================================
# LLM Mocking Fixtures
# =============================================================================

@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    from tests.mocks.llm_responses import (
        MOCK_CLASSIFICATION_RESPONSE,
        MOCK_REPORT_RESPONSE,
    )
    
    mock_client = MagicMock()
    
    # Mock chat completions
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=MOCK_REPORT_RESPONSE))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content=MOCK_REPORT_RESPONSE)
        mock_chat.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_structured_llm():
    """Mock LLM with structured output."""
    from tests.mocks.llm_responses import get_mock_classifications
    from radar.schemas import ArticleClassificationBatch
    
    mock_llm = MagicMock()
    
    def mock_invoke(messages):
        # Return mock classifications
        return get_mock_classifications(1)
    
    mock_llm.invoke = mock_invoke
    
    with patch.object(
        MagicMock, "with_structured_output", return_value=mock_llm
    ):
        yield mock_llm


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_article():
    """Return a sample article dictionary."""
    return {
        "competitor_id": "netflix",
        "source_label": "test_feed",
        "title": "Netflix Launches New Feature",
        "url": "https://example.com/netflix-news",
        "published_at": "2024-12-19T12:00:00Z",
        "raw_snippet": "Netflix announced a new streaming feature today...",
        "hash": "abc123def456",
    }


@pytest.fixture
def sample_articles():
    """Return a list of sample articles."""
    return [
        {
            "competitor_id": "netflix",
            "source_label": "variety",
            "title": "Netflix Expands Ad Tier",
            "url": "https://example.com/netflix-1",
            "published_at": "2024-12-19T12:00:00Z",
            "raw_snippet": "Netflix is expanding its ad-supported tier...",
            "hash": "hash1",
        },
        {
            "competitor_id": "disney",
            "source_label": "deadline",
            "title": "Disney+ Adds New Features",
            "url": "https://example.com/disney-1",
            "published_at": "2024-12-19T11:00:00Z",
            "raw_snippet": "Disney+ announced new streaming features...",
            "hash": "hash2",
        },
        {
            "competitor_id": "youtube",
            "source_label": "techcrunch",
            "title": "YouTube Updates Creator Tools",
            "url": "https://example.com/youtube-1",
            "published_at": "2024-12-19T10:00:00Z",
            "raw_snippet": "YouTube is rolling out new creator tools...",
            "hash": "hash3",
        },
    ]


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires API keys)",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return
    
    skip_integration = pytest.mark.skip(
        reason="Need --run-integration option to run"
    )
    
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

