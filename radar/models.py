"""
SQLAlchemy models for Tubi Radar database schema.

Tables:
- runs: Track each radar execution
- articles: Raw source items from RSS/search
- intel: LLM-understood and classified items
- annotations: Domain agent commentary
- reports: Generated report metadata
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    Index,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Run(Base):
    """Track each radar execution run."""
    
    __tablename__ = "runs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, success, error
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    articles: Mapped[list["Article"]] = relationship("Article", back_populates="run")
    reports: Mapped[list["Report"]] = relationship("Report", back_populates="run")
    
    def __repr__(self) -> str:
        return f"<Run(id={self.id}, status={self.status}, started_at={self.started_at})>"


class Article(Base):
    """Raw source items from RSS feeds or web search."""
    
    __tablename__ = "articles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    competitor_id: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'roku', 'netflix'
    source_label: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'roku_blog', 'deadline_rss'
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    raw_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # RSS description or HTML excerpt
    hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA256 of competitor_id+title+url
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="articles")
    intel: Mapped[Optional["Intel"]] = relationship("Intel", back_populates="article", uselist=False)
    
    __table_args__ = (
        Index("idx_articles_hash", "hash"),
        Index("idx_articles_published_at", "published_at"),
        Index("idx_articles_competitor", "competitor_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Article(id={self.id}, competitor={self.competitor_id}, title={self.title[:50]}...)>"


class Intel(Base):
    """LLM-understood and classified items."""
    
    __tablename__ = "intel"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(Integer, ForeignKey("articles.id"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)  # 2-3 sentence summary
    category: Mapped[str] = mapped_column(String(20), nullable=False)  # product, content, marketing, ai_ads, pricing, noise
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-10, Tubi relevance
    impact_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-10, strategic impact
    novelty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-1, set by Memory Agent
    is_duplicate_of: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("intel.id"), nullable=True)
    entities_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of entities
    llm_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Raw JSON from LLM
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article: Mapped["Article"] = relationship("Article", back_populates="intel")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="intel")
    canonical_intel: Mapped[Optional["Intel"]] = relationship(
        "Intel", remote_side="Intel.id", foreign_keys=[is_duplicate_of]
    )
    
    __table_args__ = (
        Index("idx_intel_category_scores", "category", "impact_score", "novelty_score"),
        Index("idx_intel_article", "article_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Intel(id={self.id}, category={self.category}, impact={self.impact_score})>"


class Annotation(Base):
    """Domain agent commentary on intel items."""
    
    __tablename__ = "annotations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intel_id: Mapped[int] = mapped_column(Integer, ForeignKey("intel.id"), nullable=False)
    agent_role: Mapped[str] = mapped_column(String(50), nullable=False)  # product_agent, content_agent, etc.
    so_what: Mapped[str] = mapped_column(Text, nullable=False)  # 2-3 sentence rationale
    risk_opportunity: Mapped[str] = mapped_column(String(20), nullable=False)  # risk, opportunity, neutral
    priority: Mapped[str] = mapped_column(String(5), nullable=False)  # P0, P1, P2
    suggested_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Actionable next step
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    intel: Mapped["Intel"] = relationship("Intel", back_populates="annotations")
    
    __table_args__ = (
        Index("idx_annotations_intel", "intel_id"),
        Index("idx_annotations_agent", "agent_role"),
    )
    
    def __repr__(self) -> str:
        return f"<Annotation(id={self.id}, agent={self.agent_role}, priority={self.priority})>"


class Report(Base):
    """Generated report metadata."""
    
    __tablename__ = "reports"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)  # e.g., 'reports/radar-2025-12-19.md'
    summary_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Short section of the report
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="reports")
    
    def __repr__(self) -> str:
        return f"<Report(id={self.id}, path={self.path})>"

