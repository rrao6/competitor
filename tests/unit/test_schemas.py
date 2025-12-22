"""Tests for Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from radar.schemas import (
    ArticleClassification,
    ArticleClassificationBatch,
    DomainAnnotation,
    DomainAnnotationBatch,
    TopMove,
    ReportSection,
    ReportStructure,
    NoveltyAssessment,
    ArticleInput,
    IntelInput,
)


class TestArticleClassification:
    """Tests for ArticleClassification schema."""
    
    def test_valid_classification(self):
        """Test creating a valid classification."""
        classification = ArticleClassification(
            article_id=1,
            summary="Netflix announced a new feature.",
            category="product",
            relevance_score=7.5,
            impact_score=6.0,
            entities=["Netflix", "Streaming"],
        )
        
        assert classification.article_id == 1
        assert classification.category == "product"
        assert classification.relevance_score == 7.5
    
    def test_relevance_score_bounds(self):
        """Test relevance score validation (0-10)."""
        # Valid scores
        ArticleClassification(
            article_id=1,
            summary="Test",
            category="product",
            relevance_score=0,
            impact_score=5,
        )
        ArticleClassification(
            article_id=1,
            summary="Test",
            category="product",
            relevance_score=10,
            impact_score=5,
        )
        
        # Invalid: > 10
        with pytest.raises(ValidationError):
            ArticleClassification(
                article_id=1,
                summary="Test",
                category="product",
                relevance_score=11,
                impact_score=5,
            )
        
        # Invalid: < 0
        with pytest.raises(ValidationError):
            ArticleClassification(
                article_id=1,
                summary="Test",
                category="product",
                relevance_score=-1,
                impact_score=5,
            )
    
    def test_impact_score_bounds(self):
        """Test impact score validation (0-10)."""
        with pytest.raises(ValidationError):
            ArticleClassification(
                article_id=1,
                summary="Test",
                category="product",
                relevance_score=5,
                impact_score=15,
            )
    
    def test_valid_categories(self):
        """Test that only valid categories are accepted."""
        valid_categories = ["product", "content", "marketing", "ai_ads", "pricing", "noise"]
        
        for category in valid_categories:
            c = ArticleClassification(
                article_id=1,
                summary="Test",
                category=category,
                relevance_score=5,
                impact_score=5,
            )
            assert c.category == category
        
        with pytest.raises(ValidationError):
            ArticleClassification(
                article_id=1,
                summary="Test",
                category="invalid_category",
                relevance_score=5,
                impact_score=5,
            )
    
    def test_entities_default(self):
        """Test that entities defaults to empty list."""
        c = ArticleClassification(
            article_id=1,
            summary="Test",
            category="product",
            relevance_score=5,
            impact_score=5,
        )
        assert c.entities == []


class TestArticleClassificationBatch:
    """Tests for ArticleClassificationBatch schema."""
    
    def test_batch_creation(self):
        """Test creating a batch of classifications."""
        batch = ArticleClassificationBatch(
            classifications=[
                ArticleClassification(
                    article_id=1,
                    summary="Test 1",
                    category="product",
                    relevance_score=5,
                    impact_score=5,
                ),
                ArticleClassification(
                    article_id=2,
                    summary="Test 2",
                    category="content",
                    relevance_score=7,
                    impact_score=8,
                ),
            ]
        )
        
        assert len(batch.classifications) == 2
    
    def test_empty_batch(self):
        """Test creating an empty batch."""
        batch = ArticleClassificationBatch(classifications=[])
        assert len(batch.classifications) == 0


class TestDomainAnnotation:
    """Tests for DomainAnnotation schema."""
    
    def test_valid_annotation(self):
        """Test creating a valid annotation."""
        annotation = DomainAnnotation(
            intel_id=1,
            so_what="This matters because...",
            risk_or_opportunity="opportunity",
            priority="P1",
            suggested_action="Monitor closely",
        )
        
        assert annotation.intel_id == 1
        assert annotation.risk_or_opportunity == "opportunity"
        assert annotation.priority == "P1"
    
    def test_risk_or_opportunity_values(self):
        """Test valid risk/opportunity values."""
        valid_values = ["risk", "opportunity", "neutral"]
        
        for value in valid_values:
            a = DomainAnnotation(
                intel_id=1,
                so_what="Test",
                risk_or_opportunity=value,
                priority="P1",
            )
            assert a.risk_or_opportunity == value
        
        with pytest.raises(ValidationError):
            DomainAnnotation(
                intel_id=1,
                so_what="Test",
                risk_or_opportunity="invalid",
                priority="P1",
            )
    
    def test_priority_values(self):
        """Test valid priority values."""
        valid_priorities = ["P0", "P1", "P2"]
        
        for priority in valid_priorities:
            a = DomainAnnotation(
                intel_id=1,
                so_what="Test",
                risk_or_opportunity="neutral",
                priority=priority,
            )
            assert a.priority == priority
        
        with pytest.raises(ValidationError):
            DomainAnnotation(
                intel_id=1,
                so_what="Test",
                risk_or_opportunity="neutral",
                priority="P3",  # Invalid
            )
    
    def test_suggested_action_optional(self):
        """Test that suggested_action is optional."""
        annotation = DomainAnnotation(
            intel_id=1,
            so_what="Test",
            risk_or_opportunity="neutral",
            priority="P1",
        )
        assert annotation.suggested_action is None


class TestDomainAnnotationBatch:
    """Tests for DomainAnnotationBatch schema."""
    
    def test_batch_creation(self):
        """Test creating a batch of annotations."""
        batch = DomainAnnotationBatch(
            annotations=[
                DomainAnnotation(
                    intel_id=1,
                    so_what="Test 1",
                    risk_or_opportunity="risk",
                    priority="P0",
                ),
                DomainAnnotation(
                    intel_id=2,
                    so_what="Test 2",
                    risk_or_opportunity="opportunity",
                    priority="P1",
                ),
            ]
        )
        
        assert len(batch.annotations) == 2


class TestReportSchemas:
    """Tests for report-related schemas."""
    
    def test_top_move(self):
        """Test TopMove schema."""
        move = TopMove(
            headline="Netflix Launches New Feature",
            competitor="Netflix",
            summary="Netflix announced...",
            priority="P1",
        )
        
        assert move.competitor == "Netflix"
        assert move.priority == "P1"
    
    def test_report_section(self):
        """Test ReportSection schema."""
        section = ReportSection(
            title="Product & UX",
            items=["Item 1", "Item 2", "Item 3"],
        )
        
        assert section.title == "Product & UX"
        assert len(section.items) == 3
    
    def test_report_structure(self):
        """Test ReportStructure schema."""
        report = ReportStructure(
            date="2024-12-19",
            top_moves=[
                TopMove(
                    headline="Test Move",
                    competitor="Netflix",
                    summary="Summary",
                    priority="P1",
                )
            ],
            product_ux=ReportSection(title="Product", items=["Item"]),
            content_library=ReportSection(title="Content", items=[]),
            marketing_positioning=ReportSection(title="Marketing", items=[]),
            ai_ads_pricing=ReportSection(title="AI & Ads", items=[]),
        )
        
        assert report.date == "2024-12-19"
        assert len(report.top_moves) == 1
    
    def test_suggested_actions_default(self):
        """Test that suggested_actions defaults to empty list."""
        report = ReportStructure(
            date="2024-12-19",
            top_moves=[],
            product_ux=ReportSection(title="Product", items=[]),
            content_library=ReportSection(title="Content", items=[]),
            marketing_positioning=ReportSection(title="Marketing", items=[]),
            ai_ads_pricing=ReportSection(title="AI & Ads", items=[]),
        )
        
        assert report.suggested_actions == []


class TestNoveltyAssessment:
    """Tests for NoveltyAssessment schema."""
    
    def test_valid_assessment(self):
        """Test creating a valid novelty assessment."""
        assessment = NoveltyAssessment(
            intel_id=1,
            novelty_score=0.8,
            is_duplicate=False,
        )
        
        assert assessment.novelty_score == 0.8
        assert assessment.is_duplicate is False
    
    def test_novelty_score_bounds(self):
        """Test novelty score validation (0-1)."""
        # Valid
        NoveltyAssessment(intel_id=1, novelty_score=0, is_duplicate=False)
        NoveltyAssessment(intel_id=1, novelty_score=1, is_duplicate=False)
        
        # Invalid
        with pytest.raises(ValidationError):
            NoveltyAssessment(intel_id=1, novelty_score=1.5, is_duplicate=False)
        
        with pytest.raises(ValidationError):
            NoveltyAssessment(intel_id=1, novelty_score=-0.1, is_duplicate=False)
    
    def test_duplicate_of_id_optional(self):
        """Test that duplicate_of_id is optional."""
        assessment = NoveltyAssessment(
            intel_id=1,
            novelty_score=0.5,
            is_duplicate=True,
            duplicate_of_id=2,
        )
        
        assert assessment.duplicate_of_id == 2


class TestInputSchemas:
    """Tests for input schemas."""
    
    def test_article_input(self):
        """Test ArticleInput schema."""
        article = ArticleInput(
            id=1,
            competitor_id="netflix",
            source_label="variety",
            title="Netflix News",
            url="https://example.com/article",
            published_at="2024-12-19T12:00:00Z",
            raw_snippet="Content...",
        )
        
        assert article.id == 1
        assert article.competitor_id == "netflix"
    
    def test_article_input_optional_published_at(self):
        """Test that published_at is optional."""
        article = ArticleInput(
            id=1,
            competitor_id="netflix",
            source_label="variety",
            title="Netflix News",
            url="https://example.com/article",
            published_at=None,
            raw_snippet="Content...",
        )
        
        assert article.published_at is None
    
    def test_intel_input(self):
        """Test IntelInput schema."""
        intel = IntelInput(
            id=1,
            article_id=1,
            competitor_id="netflix",
            summary="Netflix announced...",
            category="product",
            relevance_score=7.0,
            impact_score=6.0,
            entities=["Netflix"],
        )
        
        assert intel.id == 1
        assert intel.entities == ["Netflix"]
    
    def test_intel_input_entities_default(self):
        """Test that entities defaults to empty list."""
        intel = IntelInput(
            id=1,
            article_id=1,
            competitor_id="netflix",
            summary="Summary",
            category="product",
            relevance_score=5.0,
            impact_score=5.0,
        )
        
        assert intel.entities == []

