"""
Pydantic schemas for structured LLM outputs.

These models define the JSON schemas used for OpenAI structured outputs,
ensuring reliable and consistent responses from agents.
"""
from __future__ import annotations

from typing import Optional, Literal, List
from pydantic import BaseModel, Field


# =============================================================================
# Understanding Agent Schemas
# =============================================================================

class ArticleClassification(BaseModel):
    """Classification result for a single article from the Understanding Agent."""
    
    article_id: int = Field(
        description="The database ID of the article being classified"
    )
    summary: str = Field(
        description="2-3 sentence summary from Tubi's perspective: what happened, who did it, what does it change"
    )
    category: Literal["strategic", "product", "content", "marketing", "ai_ads", "pricing", "noise"] = Field(
        description="Primary category for this intel"
    )
    relevance_score: float = Field(
        ge=0, le=10,
        description="0-10 score for relevance to Tubi's product, content, or ad strategy"
    )
    impact_score: float = Field(
        ge=0, le=10,
        description="0-10 score for strategic impact (0=trivia, 10=must-know strategic move)"
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Key entities: companies, products, platforms, shows mentioned"
    )


class ArticleClassificationBatch(BaseModel):
    """Batch of article classifications."""
    
    classifications: list[ArticleClassification] = Field(
        description="List of classifications for each article"
    )


# =============================================================================
# Domain Agent Schemas
# =============================================================================

class DomainAnnotation(BaseModel):
    """Annotation from a Domain Agent (Product/Content/Marketing/AI-Ads)."""
    
    intel_id: int = Field(
        description="The database ID of the intel item being annotated"
    )
    so_what: str = Field(
        description="2-3 sentences explaining why this matters (or not) from the domain perspective"
    )
    risk_or_opportunity: Literal["risk", "opportunity", "neutral"] = Field(
        description="Whether this represents a risk, opportunity, or is neutral for Tubi"
    )
    priority: Literal["P0", "P1", "P2"] = Field(
        description="P0=urgent/strategic, P1=important but not urgent, P2=nice-to-know"
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="Optional concrete next step for the team"
    )


class DomainAnnotationBatch(BaseModel):
    """Batch of domain annotations."""
    
    annotations: list[DomainAnnotation] = Field(
        description="List of annotations for each intel item"
    )


# =============================================================================
# Editor Agent Schemas
# =============================================================================

class TopMove(BaseModel):
    """A key move highlighted in the report."""
    
    headline: str = Field(
        description="One-line headline for this move"
    )
    competitor: str = Field(
        description="Name of the competitor making this move"
    )
    summary: str = Field(
        description="2-3 sentence summary of the move and its significance"
    )
    priority: Literal["P0", "P1", "P2"] = Field(
        description="Priority level"
    )


class ReportSection(BaseModel):
    """A section in the report."""
    
    title: str = Field(
        description="Section title"
    )
    items: list[str] = Field(
        description="Bullet points for this section"
    )


class ReportStructure(BaseModel):
    """Structured report output for the Editor Agent."""
    
    date: str = Field(
        description="Report date in YYYY-MM-DD format"
    )
    top_moves: list[TopMove] = Field(
        description="3-5 key moves Tubi should care about"
    )
    product_ux: ReportSection = Field(
        description="Product & UX section"
    )
    content_library: ReportSection = Field(
        description="Content & Library section"
    )
    marketing_positioning: ReportSection = Field(
        description="Marketing & Positioning section"
    )
    ai_ads_pricing: ReportSection = Field(
        description="AI & Ads / Pricing section"
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Optional list of actionable next steps"
    )


# =============================================================================
# Memory Agent Schemas (for optional LLM-based novelty reasoning)
# =============================================================================

class NoveltyAssessment(BaseModel):
    """Assessment of an intel item's novelty."""
    
    intel_id: int = Field(
        description="The database ID of the intel item"
    )
    novelty_score: float = Field(
        ge=0, le=1,
        description="0-1 novelty score (0=duplicate/old news, 1=completely new)"
    )
    is_duplicate: bool = Field(
        description="Whether this is a duplicate of existing intel"
    )
    duplicate_of_id: Optional[int] = Field(
        default=None,
        description="ID of the intel item this duplicates, if any"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of novelty assessment"
    )


# =============================================================================
# Input Schemas (for feeding data to agents)
# =============================================================================

class ArticleInput(BaseModel):
    """Input format for an article to be classified."""
    
    id: int
    competitor_id: str
    source_label: str
    title: str
    url: str
    published_at: Optional[str]
    raw_snippet: str


class IntelInput(BaseModel):
    """Input format for intel to be annotated by domain agents."""
    
    id: int
    article_id: int
    competitor_id: str
    summary: str
    category: str
    relevance_score: float
    impact_score: float
    entities: list[str] = Field(default_factory=list)

