"""Tests for RSS fetching functionality."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from radar.tools.rss import (
    ArticleCandidate,
    compute_article_hash,
    parse_published_date,
    fetch_feed,
    get_all_feed_configs,
)
from tests.mocks.llm_responses import MOCK_RSS_CONTENT


class TestArticleHash:
    """Tests for article hash computation."""
    
    def test_hash_is_deterministic(self):
        """Test that same inputs produce same hash."""
        hash1 = compute_article_hash("netflix", "Test Title", "https://example.com")
        hash2 = compute_article_hash("netflix", "Test Title", "https://example.com")
        
        assert hash1 == hash2
    
    def test_different_inputs_different_hash(self):
        """Test that different inputs produce different hashes."""
        hash1 = compute_article_hash("netflix", "Title 1", "https://example.com/1")
        hash2 = compute_article_hash("netflix", "Title 2", "https://example.com/2")
        
        assert hash1 != hash2
    
    def test_hash_is_sha256_length(self):
        """Test that hash is SHA256 (64 hex characters)."""
        hash_value = compute_article_hash("test", "test", "test")
        
        assert len(hash_value) == 64


class TestPublishedDateParsing:
    """Tests for date parsing from feed entries."""
    
    def test_parse_published_parsed(self):
        """Test parsing from published_parsed field."""
        entry = {
            "published_parsed": (2024, 12, 19, 12, 0, 0, 0, 0, 0)
        }
        
        result = parse_published_date(entry)
        
        assert result is not None
        assert result.year == 2024
        assert result.month == 12
        assert result.day == 19
    
    def test_parse_updated_parsed(self):
        """Test fallback to updated_parsed field."""
        entry = {
            "updated_parsed": (2024, 12, 18, 10, 0, 0, 0, 0, 0)
        }
        
        result = parse_published_date(entry)
        
        assert result is not None
        assert result.day == 18
    
    def test_parse_no_date(self):
        """Test handling of missing date."""
        entry = {}
        
        result = parse_published_date(entry)
        
        assert result is None


class TestFetchFeed:
    """Tests for RSS feed fetching."""
    
    @patch("radar.tools.rss.urllib.request.urlopen")
    def test_fetch_feed_success(self, mock_urlopen):
        """Test successful feed fetch."""
        mock_response = MagicMock()
        mock_response.read.return_value = MOCK_RSS_CONTENT.encode()
        mock_urlopen.return_value = mock_response
        
        candidates = fetch_feed(
            feed_url="https://example.com/feed.xml",
            competitor_id="test",
            source_label="test_feed",
            max_items=10,
        )
        
        assert len(candidates) >= 2  # At least 2 articles
        assert candidates[0].title == "Test Article 1: Streaming News"
        assert candidates[0].competitor_id == "test"
    
    @patch("radar.tools.rss.urllib.request.urlopen")
    def test_fetch_feed_with_filter(self, mock_urlopen):
        """Test feed fetch with keyword filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = MOCK_RSS_CONTENT.encode()
        mock_urlopen.return_value = mock_response
        
        candidates = fetch_feed(
            feed_url="https://example.com/feed.xml",
            competitor_id="test",
            source_label="test_feed",
            filter_keywords=["Netflix"],
        )
        
        # Only the Netflix article should match
        assert len(candidates) == 1
        assert "Netflix" in candidates[0].title
    
    @patch("radar.tools.rss.urllib.request.urlopen")
    def test_fetch_feed_timeout(self, mock_urlopen):
        """Test handling of timeout."""
        import socket
        mock_urlopen.side_effect = socket.timeout()
        
        candidates = fetch_feed(
            feed_url="https://example.com/feed.xml",
            competitor_id="test",
            source_label="test_feed",
        )
        
        assert candidates == []
    
    @patch("radar.tools.rss.urllib.request.urlopen")
    def test_fetch_feed_url_error(self, mock_urlopen):
        """Test handling of URL error."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Not found")
        
        candidates = fetch_feed(
            feed_url="https://example.com/feed.xml",
            competitor_id="test",
            source_label="test_feed",
        )
        
        assert candidates == []


class TestArticleCandidate:
    """Tests for ArticleCandidate dataclass."""
    
    def test_create_candidate(self):
        """Test creating an article candidate."""
        candidate = ArticleCandidate(
            competitor_id="netflix",
            source_label="variety",
            title="Test Article",
            url="https://example.com/test",
            published_at=datetime.now(timezone.utc),
            raw_snippet="Test snippet",
            hash="abc123",
        )
        
        assert candidate.competitor_id == "netflix"
        assert candidate.title == "Test Article"

