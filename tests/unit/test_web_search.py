"""Tests for web search functionality."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from radar.tools.web_search import (
    SearchResult,
    search_web,
    search_competitor,
    search_all_competitors,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Netflix News",
            url="https://example.com/netflix",
            snippet="Netflix announced...",
            source="web_search",
        )
        
        assert result.title == "Netflix News"
        assert result.source == "web_search"


class TestSearchWeb:
    """Tests for web search function."""
    
    @patch("radar.tools.web_search.get_settings")
    def test_search_web_no_api_key(self, mock_get_settings, mock_config):
        """Test search with no API key."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        mock_get_settings.return_value = mock_settings
        
        results = search_web("test query")
        
        assert results == []
    
    @patch("radar.tools.web_search.OpenAI")
    @patch("radar.tools.web_search.get_settings")
    def test_search_web_error_handling(
        self, mock_get_settings, mock_openai, mock_config
    ):
        """Test search error handling."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_get_settings.return_value = mock_settings
        
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client
        
        results = search_web("test query")
        
        assert results == []


class TestSearchCompetitor:
    """Tests for competitor search."""
    
    @patch("radar.tools.web_search.search_web")
    def test_search_competitor_single_query(self, mock_search):
        """Test searching for a single competitor."""
        mock_search.return_value = [
            SearchResult(
                title="Netflix Update",
                url="https://example.com/1",
                snippet="Netflix news",
                source="web_search",
            )
        ]
        
        candidates = search_competitor(
            competitor_id="netflix",
            queries=["Netflix streaming"],
            max_results_per_query=3,
        )
        
        assert len(candidates) == 1
        assert candidates[0].competitor_id == "netflix"
        assert candidates[0].source_label == "web_search"
    
    @patch("radar.tools.web_search.search_web")
    def test_search_competitor_multiple_queries(self, mock_search):
        """Test searching with multiple queries."""
        mock_search.side_effect = [
            [
                SearchResult(
                    title="Netflix Update 1",
                    url="https://example.com/1",
                    snippet="News 1",
                    source="web_search",
                )
            ],
            [
                SearchResult(
                    title="Netflix Update 2",
                    url="https://example.com/2",
                    snippet="News 2",
                    source="web_search",
                )
            ],
        ]
        
        candidates = search_competitor(
            competitor_id="netflix",
            queries=["Netflix streaming", "Netflix originals"],
            max_results_per_query=3,
        )
        
        assert len(candidates) == 2
    
    @patch("radar.tools.web_search.search_web")
    def test_search_competitor_dedup_urls(self, mock_search):
        """Test that duplicate URLs are removed."""
        mock_search.side_effect = [
            [
                SearchResult(
                    title="Netflix Update",
                    url="https://example.com/same",  # Same URL
                    snippet="News",
                    source="web_search",
                )
            ],
            [
                SearchResult(
                    title="Netflix Update 2",
                    url="https://example.com/same",  # Same URL
                    snippet="News 2",
                    source="web_search",
                )
            ],
        ]
        
        candidates = search_competitor(
            competitor_id="netflix",
            queries=["query1", "query2"],
            max_results_per_query=3,
        )
        
        # Should only have 1 result due to dedup
        assert len(candidates) == 1


class TestSearchAllCompetitors:
    """Tests for searching all competitors."""
    
    @patch("radar.tools.web_search.search_competitor")
    def test_search_all_disabled(self, mock_search, mock_config):
        """Test search when disabled in config."""
        # Modify config to disable web search
        mock_config.global_config.enable_web_search = False
        
        with patch("radar.tools.web_search.get_config", return_value=mock_config):
            candidates = search_all_competitors(verbose=False)
        
        assert candidates == []
        mock_search.assert_not_called()
    
    @patch("radar.tools.web_search.search_competitor")
    def test_search_all_respects_max_searches(self, mock_search, mock_config):
        """Test that max searches limit is respected."""
        from radar.tools.rss import ArticleCandidate
        
        mock_search.return_value = [
            ArticleCandidate(
                competitor_id="test",
                source_label="web_search",
                title="Test",
                url="https://example.com",
                published_at=datetime.now(timezone.utc),
                raw_snippet="Test",
                hash="abc123",
            )
        ]
        
        # Limit to very few searches
        candidates = search_all_competitors(max_searches=1, verbose=False)
        
        # Should have limited number of search calls
        assert mock_search.call_count <= 2  # At most 2 queries


class TestWebSearchTool:
    """Tests for web search LangChain tool."""
    
    @patch("radar.tools.web_search.search_web")
    def test_web_search_tool(self, mock_search):
        """Test the LangChain tool wrapper."""
        from radar.tools.web_search import web_search_tool
        
        mock_search.return_value = [
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Snippet",
                source="web_search",
            )
        ]
        
        results = web_search_tool.invoke({"query": "test query"})
        
        assert len(results) == 1
        assert results[0]["title"] == "Test Result"

