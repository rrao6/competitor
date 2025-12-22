"""
Configuration loader for Tubi Radar.

Parses and validates config/radar.yaml using Pydantic models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, List, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "radar.yaml"


class FeedConfig(BaseModel):
    """Configuration for a single RSS feed."""
    label: str
    type: str = "rss"
    url: str
    filter_keywords: List[str] = Field(default_factory=list)


class CompetitorConfig(BaseModel):
    """Configuration for a single competitor."""
    id: str
    name: str
    category: str = "streaming"  # streaming, ctv, ai, social
    feeds: List[FeedConfig] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """OpenAI model configuration."""
    reasoning: str = "gpt-4o"
    structured: str = "gpt-4o"
    embedding: str = "text-embedding-3-small"
    web_search: str = "gpt-4o"


class TemperatureConfig(BaseModel):
    """Temperature settings for different tasks."""
    classification: float = 0.2
    analysis: float = 0.3
    report: float = 0.4


class ChromaConfig(BaseModel):
    """ChromaDB configuration."""
    collection_name: str = "intel_embeddings"
    similarity_threshold: float = 0.9


class DedupConfig(BaseModel):
    """Deduplication settings."""
    window_days: int = 30
    similarity_threshold: float = 0.85


class GlobalConfig(BaseModel):
    """Global configuration settings."""
    lookback_hours: int = 48
    max_articles_per_feed: int = 20
    feed_timeout: int = 15
    max_concurrent_feeds: int = 10
    min_relevance_score: float = 3.5
    min_impact_score: float = 3.5
    min_novelty_score: float = 0.2
    max_items_total: int = 100
    max_report_items: int = 20
    enable_web_search: bool = True
    max_web_searches: int = 15
    models: ModelConfig = Field(default_factory=ModelConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)


class TubiConfig(BaseModel):
    """Configuration for Tubi (our company) tracking."""
    id: str = "tubi"
    name: str = "Tubi"
    feeds: List[FeedConfig] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class RadarConfig(BaseModel):
    """Root configuration model for Competitor Monitor."""
    tubi: Optional[TubiConfig] = None
    competitors: List[CompetitorConfig] = Field(default_factory=list)
    industry_feeds: List[FeedConfig] = Field(default_factory=list)
    global_config: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    
    class Config:
        populate_by_name = True


class Settings(BaseSettings):
    """Environment-based settings."""
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    radar_db_path: Optional[str] = Field(default=None, env="RADAR_DB_PATH")
    radar_config_path: Optional[str] = Field(default=None, env="RADAR_CONFIG_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance (lazy loaded)
_config: Optional[RadarConfig] = None
_settings: Optional[Settings] = None


def load_config(config_path: Optional[Union[Path, str]] = None) -> RadarConfig:
    """Load and validate configuration from YAML file."""
    global _config
    
    if config_path is None:
        config_path = os.environ.get("RADAR_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    _config = RadarConfig.model_validate(raw_config)
    return _config


def get_config() -> RadarConfig:
    """Get the current configuration (loads if not already loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_settings() -> Settings:
    """Get environment settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_competitor_by_id(competitor_id: str) -> Optional[CompetitorConfig]:
    """Get a competitor configuration by ID."""
    config = get_config()
    for competitor in config.competitors:
        if competitor.id == competitor_id:
            return competitor
    return None


def get_all_feeds() -> List[Tuple[str, FeedConfig]]:
    """
    Get all feeds from all competitors and industry sources.
    
    Returns:
        List of (competitor_id or 'industry', feed_config) tuples.
    """
    config = get_config()
    feeds = []
    
    for competitor in config.competitors:
        for feed in competitor.feeds:
            feeds.append((competitor.id, feed))
    
    for feed in config.industry_feeds:
        feeds.append(("industry", feed))
    
    return feeds


def get_tubi_feeds() -> List[Tuple[str, FeedConfig]]:
    """
    Get Tubi-specific feeds for our company tracking.
    
    Returns:
        List of ('tubi', feed_config) tuples.
    """
    config = get_config()
    feeds = []
    
    if config.tubi:
        for feed in config.tubi.feeds:
            feeds.append(("tubi", feed))
    
    return feeds


def get_tubi_search_queries() -> List[str]:
    """Get Tubi-specific search queries."""
    config = get_config()
    if config.tubi:
        return config.tubi.search_queries
    return []

