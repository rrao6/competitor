"""Tests for database operations."""
from __future__ import annotations

import pytest
from datetime import datetime

from radar.database import init_database, get_session
from radar.models import Run, Article, Intel


class TestDatabaseInit:
    """Tests for database initialization."""
    
    def test_database_creates_tables(self, test_db):
        """Test that database initialization creates all tables."""
        from sqlalchemy import inspect
        from radar.database import get_engine
        
        engine = get_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert "runs" in tables
        assert "articles" in tables
        assert "intel" in tables
        assert "annotations" in tables
        assert "reports" in tables


class TestRunOperations:
    """Tests for run CRUD operations."""
    
    def test_create_run(self, test_db):
        """Test creating a new run."""
        from radar.tools.db_tools import create_run, get_run
        
        run_id = create_run()
        
        assert run_id > 0
        
        run = get_run(run_id)
        assert run is not None
        assert run["status"] == "running"
    
    def test_complete_run(self, test_db):
        """Test completing a run."""
        from radar.tools.db_tools import create_run, complete_run, get_run
        
        run_id = create_run()
        complete_run(run_id, status="success", notes="Test complete")
        
        run = get_run(run_id)
        assert run["status"] == "success"
        assert run["notes"] == "Test complete"
        assert run["finished_at"] is not None


class TestArticleOperations:
    """Tests for article CRUD operations."""
    
    def test_store_articles(self, test_db, sample_article):
        """Test storing articles."""
        from radar.tools.db_tools import create_run, store_articles
        
        run_id = create_run()
        count = store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        assert count == 1
    
    def test_dedup_articles(self, test_db, sample_article):
        """Test that duplicate articles are not stored."""
        from radar.tools.db_tools import create_run, store_articles
        
        run_id = create_run()
        
        # Store once
        count1 = store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        # Try to store again
        count2 = store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        assert count1 == 1
        assert count2 == 0  # Duplicate should not be stored
    
    def test_get_unprocessed_articles(self, test_db, sample_article):
        """Test getting unprocessed articles."""
        from radar.tools.db_tools import (
            create_run, store_articles, get_unprocessed_articles
        )
        
        run_id = create_run()
        store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        unprocessed = get_unprocessed_articles.invoke({
            "run_id": run_id,
            "limit": 10,
        })
        
        assert len(unprocessed) == 1
        assert unprocessed[0]["title"] == sample_article["title"]


class TestIntelOperations:
    """Tests for intel CRUD operations."""
    
    def test_store_intel(self, test_db, sample_article):
        """Test storing intel records."""
        from radar.tools.db_tools import (
            create_run, store_articles, store_intel, get_unprocessed_articles
        )
        
        run_id = create_run()
        store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        articles = get_unprocessed_articles.invoke({"run_id": run_id, "limit": 1})
        article_id = articles[0]["id"]
        
        intel_count = store_intel.invoke({
            "records": [{
                "article_id": article_id,
                "summary": "Test summary",
                "category": "product",
                "relevance_score": 7.0,
                "impact_score": 6.0,
                "entities": ["Netflix"],
            }]
        })
        
        assert intel_count == 1
    
    def test_get_all_intel_for_run(self, test_db, sample_article):
        """Test getting all intel for a run."""
        from radar.tools.db_tools import (
            create_run, store_articles, store_intel, 
            get_unprocessed_articles, get_all_intel_for_run
        )
        
        run_id = create_run()
        store_articles.invoke({"run_id": run_id, "items": [sample_article]})
        
        articles = get_unprocessed_articles.invoke({"run_id": run_id, "limit": 1})
        article_id = articles[0]["id"]
        
        store_intel.invoke({
            "records": [{
                "article_id": article_id,
                "summary": "Test summary for report",
                "category": "product",
                "relevance_score": 8.0,
                "impact_score": 7.0,
                "entities": ["Netflix"],
            }]
        })
        
        intel_list = get_all_intel_for_run(
            run_id=run_id,
            min_relevance=4.0,
            min_impact=4.0,
        )
        
        assert len(intel_list) == 1
        assert intel_list[0]["summary"] == "Test summary for report"

