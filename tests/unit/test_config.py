"""Tests for configuration loading and validation."""
from __future__ import annotations

import pytest
from pathlib import Path

from radar.config import (
    load_config,
    RadarConfig,
    CompetitorConfig,
    FeedConfig,
    GlobalConfig,
)


class TestConfigLoading:
    """Tests for config file loading."""
    
    def test_load_config_from_file(self, test_config_file):
        """Test loading config from a YAML file."""
        config = load_config(test_config_file)
        
        assert isinstance(config, RadarConfig)
        assert len(config.competitors) == 1
        assert config.competitors[0].id == "test_competitor"
    
    def test_config_has_required_fields(self, test_config_file):
        """Test that loaded config has all required fields."""
        config = load_config(test_config_file)
        
        assert hasattr(config, "competitors")
        assert hasattr(config, "industry_feeds")
        assert hasattr(config, "global_config")
    
    def test_config_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


class TestCompetitorConfig:
    """Tests for competitor configuration."""
    
    def test_competitor_with_feeds(self, test_config_file):
        """Test competitor with RSS feeds."""
        config = load_config(test_config_file)
        competitor = config.competitors[0]
        
        assert competitor.id == "test_competitor"
        assert competitor.name == "Test Competitor"
        assert len(competitor.feeds) == 1
        assert competitor.feeds[0].label == "test_feed"
    
    def test_competitor_with_search_queries(self, test_config_file):
        """Test competitor with search queries."""
        config = load_config(test_config_file)
        competitor = config.competitors[0]
        
        assert len(competitor.search_queries) == 1
        assert "test query" in competitor.search_queries


class TestGlobalConfig:
    """Tests for global configuration settings."""
    
    def test_default_values(self):
        """Test default values for global config."""
        config = GlobalConfig()
        
        assert config.lookback_hours == 48
        assert config.max_articles_per_feed == 20
        assert config.enable_web_search is True
    
    def test_model_config(self, test_config_file):
        """Test model configuration."""
        config = load_config(test_config_file)
        
        assert config.global_config.models.structured == "gpt-4o"
        assert config.global_config.models.embedding == "text-embedding-3-small"


class TestFeedConfig:
    """Tests for feed configuration."""
    
    def test_feed_with_filter_keywords(self, test_config_file):
        """Test industry feed with filter keywords."""
        config = load_config(test_config_file)
        industry_feed = config.industry_feeds[0]
        
        assert industry_feed.label == "test_industry"
        assert "streaming" in industry_feed.filter_keywords
        assert "video" in industry_feed.filter_keywords

