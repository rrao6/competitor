"""Integration tests for the full Tubi Radar pipeline.

These tests require:
- OPENAI_API_KEY environment variable
- Network access to RSS feeds

Run with: pytest tests/integration/ --run-integration
"""
from __future__ import annotations

import os
import pytest
from pathlib import Path


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_config_loads_from_real_file(self):
        """Test loading the actual config file."""
        from radar.config import load_config
        
        config_path = Path(__file__).parent.parent.parent / "config" / "radar.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        config = load_config(config_path)
        
        assert len(config.competitors) >= 20  # Should have many competitors
        assert len(config.industry_feeds) >= 15  # Should have many feeds
    
    def test_config_has_expected_competitors(self):
        """Test that expected competitors are configured."""
        from radar.config import load_config
        
        config_path = Path(__file__).parent.parent.parent / "config" / "radar.yaml"
        config = load_config(config_path)
        
        competitor_ids = [c.id for c in config.competitors]
        
        expected = ["netflix", "disney", "youtube", "amazon", "roku"]
        for comp in expected:
            assert comp in competitor_ids, f"Missing competitor: {comp}"


@pytest.mark.integration
class TestRSSIntegration:
    """Integration tests for RSS fetching."""
    
    def test_fetch_single_feed(self):
        """Test fetching from a single real RSS feed."""
        from radar.tools.rss import fetch_feed
        
        # Use a reliable feed for testing
        candidates = fetch_feed(
            feed_url="https://blog.youtube/rss/",
            competitor_id="youtube",
            source_label="youtube_blog",
            max_items=5,
            timeout=15,
        )
        
        assert len(candidates) > 0
        assert candidates[0].competitor_id == "youtube"
        assert candidates[0].title is not None
        assert candidates[0].url is not None
    
    def test_fetch_feed_with_filter(self):
        """Test fetching with keyword filter."""
        from radar.tools.rss import fetch_feed
        
        candidates = fetch_feed(
            feed_url="https://www.theverge.com/rss/index.xml",
            competitor_id="industry",
            source_label="the_verge",
            max_items=20,
            filter_keywords=["streaming", "Netflix", "YouTube"],
            timeout=15,
        )
        
        # Should have some articles or none (depending on current news)
        assert candidates is not None
    
    def test_parallel_fetch_multiple_feeds(self):
        """Test parallel fetching of multiple feeds."""
        from radar.tools.rss import fetch_all_feeds_parallel
        
        candidates = fetch_all_feeds_parallel(max_workers=5, verbose=False)
        
        assert len(candidates) > 0
        
        # Check we got articles from different sources
        sources = set(c.source_label for c in candidates)
        assert len(sources) >= 1


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_full_crud_cycle(self, test_db):
        """Test complete CRUD cycle for articles and intel."""
        from radar.tools.db_tools import (
            create_run, complete_run, get_run,
            store_articles, get_unprocessed_articles,
            store_intel, get_all_intel_for_run,
        )
        
        # Create run
        run_id = create_run()
        assert run_id > 0
        
        run = get_run(run_id)
        assert run["status"] == "running"
        
        # Store article
        articles = [{
            "competitor_id": "netflix",
            "source_label": "test",
            "title": "Integration Test Article",
            "url": "https://example.com/integration-test-unique",
            "published_at": "2024-12-19T12:00:00Z",
            "raw_snippet": "Test content",
            "hash": "integration_unique_hash_12345",
        }]
        count = store_articles.invoke({"run_id": run_id, "items": articles})
        assert count == 1
        
        # Get unprocessed
        unprocessed = get_unprocessed_articles.invoke({"run_id": run_id, "limit": 10})
        assert len(unprocessed) == 1
        article_id = unprocessed[0]["id"]
        
        # Store intel
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
        
        # Get intel for report
        intel_list = get_all_intel_for_run(run_id, min_relevance=4.0, min_impact=4.0)
        assert len(intel_list) == 1
        
        # Complete run
        complete_run(run_id, status="success", notes="Integration test")
        run = get_run(run_id)
        assert run["status"] == "success"
    
    def test_article_deduplication(self, test_db):
        """Test that duplicate articles are not stored."""
        from radar.tools.db_tools import create_run, store_articles
        
        run_id = create_run()
        
        article = {
            "competitor_id": "netflix",
            "source_label": "test",
            "title": "Dedup Test",
            "url": "https://example.com/dedup-test",
            "raw_snippet": "Content",
            "hash": "dedup_test_hash_xyz",
        }
        
        # First store
        count1 = store_articles.invoke({"run_id": run_id, "items": [article]})
        assert count1 == 1
        
        # Second store (same hash)
        count2 = store_articles.invoke({"run_id": run_id, "items": [article]})
        assert count2 == 0


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests that require OpenAI API."""
    
    def test_understanding_agent_real_llm(self, test_db):
        """Test Understanding Agent with real LLM calls."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        from radar.tools.db_tools import create_run, store_articles
        from radar.agents.understanding import UnderstandingAgent
        
        run_id = create_run()
        store_articles.invoke({
            "run_id": run_id,
            "items": [{
                "competitor_id": "netflix",
                "source_label": "test",
                "title": "Netflix Announces Major Streaming Innovation",
                "url": "https://example.com/netflix-llm-test",
                "published_at": "2024-12-19T12:00:00Z",
                "raw_snippet": "Netflix today announced a breakthrough in streaming technology that will enable 8K streaming at reduced bandwidth. This development leverages new AI-powered compression.",
                "hash": "llm_integration_test_unique_hash",
            }],
        })
        
        agent = UnderstandingAgent(batch_size=1)
        result = agent.run(run_id=run_id, index_embeddings=False)
        
        assert result["intel_created"] == 1
    
    def test_editor_agent_real_llm(self, test_db):
        """Test Editor Agent with real LLM calls."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        from radar.tools.db_tools import (
            create_run, store_articles, store_intel, get_unprocessed_articles
        )
        from radar.agents.editor import EditorAgent
        
        run_id = create_run()
        
        # Store article
        store_articles.invoke({
            "run_id": run_id,
            "items": [{
                "competitor_id": "netflix",
                "source_label": "test",
                "title": "Editor Test Article",
                "url": "https://example.com/editor-test",
                "raw_snippet": "Content",
                "hash": "editor_test_hash",
            }],
        })
        
        # Get article ID
        articles = get_unprocessed_articles.invoke({"run_id": run_id, "limit": 1})
        article_id = articles[0]["id"]
        
        # Store intel
        store_intel.invoke({
            "records": [{
                "article_id": article_id,
                "summary": "Netflix announced a major update to its streaming platform.",
                "category": "product",
                "relevance_score": 8.0,
                "impact_score": 7.0,
                "entities": ["Netflix"],
            }]
        })
        
        agent = EditorAgent()
        result = agent.run(run_id=run_id)
        
        assert result["intel_items_included"] == 1
        assert result["report_path"] is not None
        assert result["report_length"] > 0


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_phase1_pipeline(self, test_db):
        """Test Phase 1 pipeline execution."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        from run_radar import run_phase1
        
        result = run_phase1(verbose=False)
        
        assert "run_id" in result
        assert result["status"] in ["success", "no_articles", "no_intel"]
    
    def test_quick_mode_pipeline(self, test_db):
        """Test quick mode pipeline."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        from run_radar import run_full_pipeline
        
        result = run_full_pipeline(
            enable_web_search=False,
            enable_memory=False,
            enable_domain_agents=False,
        )
        
        assert "run_id" in result
        assert result["status"] in ["success", "error"]


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for vector store."""
    
    def test_embed_and_search(self):
        """Test embedding and searching."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        from radar.tools.vector import (
            embed_and_index_intel,
            search_similar_intel,
            reset_vector_store,
        )
        
        # Reset to clean state
        reset_vector_store()
        
        # Embed some items
        embed_and_index_intel.invoke({
            "intel_id": 1,
            "text": "Netflix announced new streaming features for ad-supported tier.",
            "metadata": {"category": "product"},
        })
        
        embed_and_index_intel.invoke({
            "intel_id": 2,
            "text": "Disney Plus launched new original content series.",
            "metadata": {"category": "content"},
        })
        
        # Search
        results = search_similar_intel.invoke({
            "text": "Netflix streaming platform update",
            "top_k": 5,
        })
        
        assert len(results) > 0
        # Netflix item should be most similar
        assert results[0]["intel_id"] == 1
