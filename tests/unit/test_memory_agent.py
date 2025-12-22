"""Tests for Memory Agent."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from radar.agents.memory import MemoryAgent


class TestMemoryAgentNoveltySimple:
    """Tests for simple novelty computation."""
    
    def test_novelty_url_duplicate(self, mock_config):
        """Test URL-based duplicate detection."""
        agent = MemoryAgent()
        
        existing_intel = [
            {"id": 1, "url": "https://example.com/article1", "summary": "Article one"},
            {"id": 2, "url": "https://example.com/article2", "summary": "Article two"},
        ]
        
        result = agent._compute_novelty_simple(
            intel_id=3,
            summary="Some new summary",
            url="https://example.com/article1",  # Duplicate URL
            existing_intel=existing_intel,
        )
        
        assert result["novelty_score"] == 0.0
        assert result["is_duplicate_of"] == 1
    
    def test_novelty_high_text_similarity(self, mock_config):
        """Test text-based duplicate detection."""
        agent = MemoryAgent()
        
        existing_intel = [
            {
                "id": 1,
                "url": "https://example.com/article1",
                "summary": "Netflix announced streaming features today morning",
            },
        ]
        
        result = agent._compute_novelty_simple(
            intel_id=2,
            summary="Netflix announced streaming features today morning update",  # Very similar
            url="https://example.com/different",
            existing_intel=existing_intel,
        )
        
        assert result["novelty_score"] == 0.1
        assert result["is_duplicate_of"] == 1
    
    def test_novelty_unique_content(self, mock_config):
        """Test novelty score for unique content."""
        agent = MemoryAgent()
        
        existing_intel = [
            {"id": 1, "url": "https://example.com/article1", "summary": "Netflix news"},
        ]
        
        result = agent._compute_novelty_simple(
            intel_id=2,
            summary="Disney announced completely different features",
            url="https://example.com/different",
            existing_intel=existing_intel,
        )
        
        assert result["novelty_score"] == 1.0
        assert result["is_duplicate_of"] is None
    
    def test_novelty_moderately_similar(self, mock_config):
        """Test novelty when there are some similar items."""
        agent = MemoryAgent()
        
        # Create several items about streaming
        existing_intel = [
            {"id": 1, "url": "https://ex.com/1", "summary": "Netflix streaming news today"},
            {"id": 2, "url": "https://ex.com/2", "summary": "Netflix streaming update yesterday"},
            {"id": 3, "url": "https://ex.com/3", "summary": "Netflix streaming features new"},
        ]
        
        result = agent._compute_novelty_simple(
            intel_id=4,
            summary="Netflix streaming announcement latest",
            url="https://example.com/different",
            existing_intel=existing_intel,
        )
        
        # Should have moderate novelty (some similar items exist)
        assert 0.3 <= result["novelty_score"] <= 0.8
    
    def test_novelty_skip_self(self, mock_config):
        """Test that an item doesn't compare to itself."""
        agent = MemoryAgent()
        
        existing_intel = [
            {"id": 1, "url": "https://example.com/article1", "summary": "Exact match summary"},
        ]
        
        result = agent._compute_novelty_simple(
            intel_id=1,  # Same ID as existing
            summary="Exact match summary",
            url="https://different.com",
            existing_intel=existing_intel,
        )
        
        # Should not mark as duplicate of itself
        assert result["is_duplicate_of"] is None


class TestMemoryAgentRun:
    """Tests for Memory Agent run method."""
    
    @patch("radar.agents.memory.get_recent_intel_for_dedup")
    @patch("radar.agents.memory.store_novelty_scores")
    def test_run_no_intel(self, mock_store, mock_get_intel, mock_config):
        """Test run with no intel to process."""
        mock_get_intel.invoke.return_value = []
        
        agent = MemoryAgent()
        result = agent.run(run_id=1, use_vector_search=False)
        
        assert result["processed"] == 0
        assert result["duplicates_found"] == 0
    
    @patch("radar.agents.memory.get_recent_intel_for_dedup")
    @patch("radar.agents.memory.store_novelty_scores")
    def test_run_with_new_intel(self, mock_store, mock_get_intel, mock_config):
        """Test run with new intel items."""
        mock_get_intel.invoke.return_value = [
            {
                "id": 1,
                "url": "https://example.com/1",
                "summary": "Netflix news",
                "category": "product",
                "novelty_score": None,  # New item
            },
            {
                "id": 2,
                "url": "https://example.com/2",
                "summary": "Disney update",
                "category": "content",
                "novelty_score": None,  # New item
            },
        ]
        
        agent = MemoryAgent()
        result = agent.run(run_id=1, use_vector_search=False)
        
        assert result["processed"] == 2
        mock_store.invoke.assert_called_once()
    
    @patch("radar.agents.memory.get_recent_intel_for_dedup")
    @patch("radar.agents.memory.store_novelty_scores")
    def test_run_finds_duplicates(self, mock_store, mock_get_intel, mock_config):
        """Test run that finds duplicates."""
        mock_get_intel.invoke.return_value = [
            {
                "id": 1,
                "url": "https://example.com/article",
                "summary": "Original article",
                "category": "product",
                "novelty_score": 0.9,  # Already processed
            },
            {
                "id": 2,
                "url": "https://example.com/article",  # Same URL = duplicate
                "summary": "Duplicate article",
                "category": "product",
                "novelty_score": None,  # New item
            },
        ]
        
        agent = MemoryAgent()
        result = agent.run(run_id=1, use_vector_search=False)
        
        assert result["duplicates_found"] == 1
    
    @patch("radar.agents.memory.get_recent_intel_for_dedup")
    @patch("radar.agents.memory.store_novelty_scores")
    def test_run_all_already_processed(self, mock_store, mock_get_intel, mock_config):
        """Test run when all items already have novelty scores."""
        mock_get_intel.invoke.return_value = [
            {
                "id": 1,
                "url": "https://example.com/1",
                "summary": "Already processed",
                "category": "product",
                "novelty_score": 0.8,  # Already has score
            },
        ]
        
        agent = MemoryAgent()
        result = agent.run(run_id=1, use_vector_search=False)
        
        assert result["processed"] == 0
        mock_store.invoke.assert_not_called()


class TestMemoryAgentVectorSearch:
    """Tests for vector-based novelty computation."""
    
    @patch("radar.tools.vector.find_duplicates")
    @patch("radar.tools.vector.search_similar_intel")
    def test_novelty_vector_duplicate_found(
        self, mock_search, mock_find_dupes, mock_config
    ):
        """Test vector-based duplicate detection."""
        mock_find_dupes.return_value = [
            {"intel_id": 1, "similarity": 0.95},
        ]
        
        agent = MemoryAgent()
        result = agent._compute_novelty_vector(
            intel_id=2,
            summary="Test summary",
            url="https://example.com",
        )
        
        assert result["novelty_score"] == 0.0
        assert result["is_duplicate_of"] == 1
    
    @patch("radar.tools.vector.find_duplicates")
    @patch("radar.tools.vector.search_similar_intel")
    def test_novelty_vector_no_duplicates(
        self, mock_search, mock_find_dupes, mock_config
    ):
        """Test vector novelty when no duplicates found."""
        mock_find_dupes.return_value = []
        mock_search.invoke.return_value = []
        
        agent = MemoryAgent()
        result = agent._compute_novelty_vector(
            intel_id=1,
            summary="Unique content",
            url="https://example.com",
        )
        
        assert result["novelty_score"] == 1.0
        assert result["is_duplicate_of"] is None
    
    @patch("radar.tools.vector.find_duplicates")
    @patch("radar.tools.vector.search_similar_intel")
    def test_novelty_vector_similar_items(
        self, mock_search, mock_find_dupes, mock_config
    ):
        """Test vector novelty with similar but not duplicate items."""
        mock_find_dupes.return_value = []
        mock_search.invoke.return_value = [
            {"intel_id": 2, "similarity": 0.6, "document": "Similar", "metadata": {}},
            {"intel_id": 3, "similarity": 0.55, "document": "Similar", "metadata": {}},
        ]
        
        agent = MemoryAgent()
        result = agent._compute_novelty_vector(
            intel_id=1,
            summary="Somewhat unique",
            url="https://example.com",
        )
        
        # Should have reduced novelty due to similar items
        assert 0.1 < result["novelty_score"] < 1.0

