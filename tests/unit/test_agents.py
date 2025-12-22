"""Tests for agent functionality with LLM mocking."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from radar.agents.base import BaseAgent
from radar.agents.ingestion import IngestionAgent
from radar.agents.understanding import UnderstandingAgent
from radar.agents.editor import EditorAgent
from radar.schemas import ArticleClassificationBatch
from tests.mocks.llm_responses import (
    get_mock_classifications,
    MOCK_REPORT_RESPONSE,
)


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_agent_has_role(self, mock_config):
        """Test that agents have defined roles."""
        agent = IngestionAgent()
        
        assert agent.agent_role == "ingestion_agent"
    
    def test_agent_repr(self, mock_config):
        """Test agent string representation."""
        agent = IngestionAgent()
        repr_str = repr(agent)
        
        assert "IngestionAgent" in repr_str
        assert "ingestion_agent" in repr_str


class TestIngestionAgent:
    """Tests for Ingestion Agent."""
    
    @patch("radar.agents.ingestion.fetch_all_feeds")
    @patch("radar.agents.ingestion.store_articles_batch")
    def test_ingestion_run(self, mock_store, mock_fetch, mock_config, test_db):
        """Test ingestion agent run."""
        from radar.tools.rss import ArticleCandidate
        from datetime import datetime, timezone
        
        # Mock feed results
        mock_fetch.return_value = [
            ArticleCandidate(
                competitor_id="netflix",
                source_label="test",
                title="Test Article",
                url="https://example.com/test",
                published_at=datetime.now(timezone.utc),
                raw_snippet="Test content",
                hash="abc123",
            )
        ]
        mock_store.return_value = 1
        
        agent = IngestionAgent()
        result = agent.run(run_id=1, enable_web_search=False)
        
        assert result["candidates_found"] == 1
        assert mock_store.called


class TestUnderstandingAgent:
    """Tests for Understanding Agent with mocked LLM."""
    
    @patch("radar.agents.understanding.get_unprocessed_articles")
    @patch("radar.agents.understanding.store_intel_from_classifications")
    def test_understanding_no_articles(
        self, mock_store, mock_get_articles, mock_config, test_db
    ):
        """Test understanding agent with no articles."""
        mock_get_articles.invoke = MagicMock(return_value=[])
        
        agent = UnderstandingAgent()
        result = agent.run(run_id=1, index_embeddings=False)
        
        assert result["articles_processed"] == 0
        assert result["intel_created"] == 0
    
    @patch("radar.agents.understanding.get_unprocessed_articles")
    @patch("radar.agents.understanding.store_intel_from_classifications")
    def test_understanding_with_articles(
        self, mock_store, mock_get_articles, mock_config, test_db, sample_articles
    ):
        """Test understanding agent with articles."""
        mock_get_articles.invoke = MagicMock(return_value=sample_articles)
        mock_store.return_value = len(sample_articles)
        
        # Mock the LLM structured output
        with patch.object(UnderstandingAgent, "_classify_batch") as mock_classify:
            mock_classify.return_value = get_mock_classifications(len(sample_articles)).classifications
            
            agent = UnderstandingAgent()
            result = agent.run(run_id=1, index_embeddings=False)
            
            assert result["articles_processed"] == len(sample_articles)


class TestEditorAgent:
    """Tests for Editor Agent with mocked LLM."""
    
    @patch("radar.agents.editor.get_all_intel_for_run")
    @patch("radar.agents.editor.create_report_file")
    def test_editor_generates_report(
        self, mock_create_file, mock_get_intel, mock_config, test_db
    ):
        """Test editor agent report generation."""
        mock_get_intel.return_value = [
            {
                "id": 1,
                "article_id": 1,
                "competitor_id": "netflix",
                "title": "Test Article",
                "url": "https://example.com",
                "summary": "Test summary",
                "category": "product",
                "relevance_score": 8.0,
                "impact_score": 7.0,
                "novelty_score": 0.8,
                "entities": ["Netflix"],
                "annotations": [],
            }
        ]
        mock_create_file.invoke = MagicMock(return_value="/reports/test.md")
        
        # Mock the LLM
        with patch.object(EditorAgent, "get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content=MOCK_REPORT_RESPONSE)
            mock_get_llm.return_value = mock_llm
            
            agent = EditorAgent()
            result = agent.run(run_id=1)
            
            assert "report_path" in result
            assert result["intel_items_included"] == 1
    
    @patch("radar.agents.editor.get_all_intel_for_run")
    @patch("radar.agents.editor.create_report_file")
    def test_editor_empty_intel(
        self, mock_create_file, mock_get_intel, mock_config, test_db
    ):
        """Test editor with no intel items."""
        mock_get_intel.return_value = []
        mock_create_file.invoke = MagicMock(return_value="/reports/empty.md")
        
        agent = EditorAgent()
        result = agent.run(run_id=1)
        
        assert result["intel_items_included"] == 0

