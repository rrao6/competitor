"""
Mock LLM responses for testing.

Provides canned responses to avoid API calls during tests.
"""
from __future__ import annotations

from radar.schemas import (
    ArticleClassification,
    ArticleClassificationBatch,
    DomainAnnotation,
    DomainAnnotationBatch,
    TopMove,
    ReportSection,
    ReportStructure,
    NoveltyAssessment,
)


# =============================================================================
# Mock Classification Responses
# =============================================================================

def get_mock_classification(
    article_id: int = 1,
    category: str = "product",
    relevance: float = 7.5,
    impact: float = 6.0,
) -> ArticleClassification:
    """Generate a mock classification for testing."""
    return ArticleClassification(
        article_id=article_id,
        summary=f"This is a test summary for article {article_id} about competitive moves.",
        category=category,
        relevance_score=relevance,
        impact_score=impact,
        entities=["Netflix", "Streaming", "AVOD"],
    )


def get_mock_classifications(
    count: int = 3,
    category: str = "product",
) -> ArticleClassificationBatch:
    """Generate a batch of mock classifications."""
    return ArticleClassificationBatch(
        classifications=[
            get_mock_classification(
                article_id=i,
                category=category,
                relevance=7.0 + (i % 3),
                impact=6.0 + (i % 4),
            )
            for i in range(1, count + 1)
        ]
    )


MOCK_CLASSIFICATION_RESPONSE = {
    "classifications": [
        {
            "article_id": 1,
            "summary": "Netflix announced expansion of its ad-supported tier with new features.",
            "category": "product",
            "relevance_score": 8.0,
            "impact_score": 7.0,
            "entities": ["Netflix", "AVOD", "Advertising"],
        },
        {
            "article_id": 2,
            "summary": "Disney+ is adding new personalization features to compete with Netflix.",
            "category": "product",
            "relevance_score": 7.5,
            "impact_score": 6.5,
            "entities": ["Disney+", "Personalization"],
        },
        {
            "article_id": 3,
            "summary": "Roku announced new original content for The Roku Channel.",
            "category": "content",
            "relevance_score": 6.5,
            "impact_score": 5.5,
            "entities": ["Roku", "Originals"],
        },
    ]
}


# =============================================================================
# Mock Domain Annotation Responses
# =============================================================================

def get_mock_annotation(
    intel_id: int = 1,
    risk_or_opportunity: str = "opportunity",
    priority: str = "P1",
) -> DomainAnnotation:
    """Generate a mock domain annotation."""
    return DomainAnnotation(
        intel_id=intel_id,
        so_what=f"This development could affect Tubi's position in the AVOD market. Intel #{intel_id} analysis.",
        risk_or_opportunity=risk_or_opportunity,
        priority=priority,
        suggested_action="Monitor feature adoption and consider similar implementation.",
    )


def get_mock_annotations(
    count: int = 3,
    risk_or_opportunity: str = "opportunity",
) -> DomainAnnotationBatch:
    """Generate a batch of mock annotations."""
    priorities = ["P0", "P1", "P2"]
    return DomainAnnotationBatch(
        annotations=[
            get_mock_annotation(
                intel_id=i,
                risk_or_opportunity=risk_or_opportunity,
                priority=priorities[i % 3],
            )
            for i in range(1, count + 1)
        ]
    )


# =============================================================================
# Mock Report Response
# =============================================================================

MOCK_REPORT_RESPONSE = """# Tubi Radar: Competitive Intelligence Digest
**Date:** 2024-12-19

---

## Top Moves

### 1. Netflix Expands Ad-Supported Tier
Netflix is expanding its ad-supported tier with new features and improved targeting.

**So What:** This increases competition for Tubi in the AVOD space.

### 2. Disney+ Launches Personalization Features
Disney+ announced new AI-powered personalization features.

**So What:** Sets new expectations for streaming UX.

---

## Product & UX

- **Netflix Ad Tier Expansion:** New features for advertisers including improved targeting.
- **Disney+ Personalization:** Enhanced recommendation engine using machine learning.
- **Roku OS Update:** New operating system features for smart TVs.

---

## Content & Library

- **Roku Originals:** New original content slate announced for 2025.
- **Peacock Sports:** Exclusive sports content deals.

---

## Marketing & Positioning

- **Netflix Brand Campaign:** Major marketing push for ad-supported tier.
- **YouTube Premium Push:** Increased promotion of subscription offering.

---

## AI & Ads / Pricing

- Netflix improving ad targeting capabilities with AI.
- Google testing new CTV ad formats.
- Roku launching new ad measurement tools.

---

## Suggested Actions

1. Monitor Netflix ad tier adoption rates and advertiser feedback.
2. Evaluate competitive feature parity for personalization.
3. Track content acquisition strategies of competitors.
4. Assess CTV advertising technology developments.

---

*Report generated by Tubi Radar*
"""


def get_mock_report_structure() -> ReportStructure:
    """Generate a mock report structure."""
    return ReportStructure(
        date="2024-12-19",
        top_moves=[
            TopMove(
                headline="Netflix Expands Ad-Supported Tier",
                competitor="Netflix",
                summary="Netflix is expanding its ad-supported tier with new targeting features.",
                priority="P1",
            ),
            TopMove(
                headline="Disney+ Launches AI Personalization",
                competitor="Disney+",
                summary="Disney+ announced AI-powered personalization improvements.",
                priority="P1",
            ),
        ],
        product_ux=ReportSection(
            title="Product & UX",
            items=[
                "Netflix: Ad tier expansion with new targeting features",
                "Disney+: Enhanced personalization engine",
                "Roku: OS update with improved content discovery",
            ],
        ),
        content_library=ReportSection(
            title="Content & Library",
            items=[
                "Roku: New original content slate for 2025",
                "Peacock: Exclusive sports content deals",
            ],
        ),
        marketing_positioning=ReportSection(
            title="Marketing & Positioning",
            items=[
                "Netflix: Major ad-supported tier campaign",
            ],
        ),
        ai_ads_pricing=ReportSection(
            title="AI & Ads / Pricing",
            items=[
                "Netflix: AI-powered ad targeting improvements",
                "Google: New CTV ad formats testing",
            ],
        ),
        suggested_actions=[
            "Monitor Netflix ad tier adoption",
            "Evaluate personalization feature parity",
            "Track competitor content strategies",
        ],
    )


# =============================================================================
# Mock Novelty Assessment
# =============================================================================

def get_mock_novelty_assessment(
    intel_id: int = 1,
    novelty_score: float = 0.8,
    is_duplicate: bool = False,
) -> NoveltyAssessment:
    """Generate a mock novelty assessment."""
    return NoveltyAssessment(
        intel_id=intel_id,
        novelty_score=novelty_score,
        is_duplicate=is_duplicate,
        duplicate_of_id=None if not is_duplicate else intel_id - 1,
        reasoning="Unique content not seen in recent intel." if not is_duplicate else "Similar to existing intel.",
    )


# =============================================================================
# Mock RSS Feed Response
# =============================================================================

MOCK_RSS_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <description>Test RSS Feed</description>
    <item>
      <title>Test Article 1: Streaming News</title>
      <link>https://example.com/article1</link>
      <description>This is a test article about streaming services.</description>
      <pubDate>Thu, 19 Dec 2024 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Test Article 2: Netflix Update</title>
      <link>https://example.com/article2</link>
      <description>Netflix announced new streaming features today.</description>
      <pubDate>Thu, 19 Dec 2024 11:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Test Article 3: Disney+ Content</title>
      <link>https://example.com/article3</link>
      <description>Disney+ adds new original series to library.</description>
      <pubDate>Thu, 19 Dec 2024 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


MOCK_ATOM_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <link href="https://example.com"/>
  <entry>
    <title>Atom Article 1</title>
    <link href="https://example.com/atom1"/>
    <summary>Atom feed test article.</summary>
    <updated>2024-12-19T12:00:00Z</updated>
  </entry>
</feed>
"""


# =============================================================================
# Mock Web Search Responses
# =============================================================================

MOCK_WEB_SEARCH_RESULTS = [
    {
        "title": "Netflix Announces New Features",
        "url": "https://news.example.com/netflix-features",
        "snippet": "Netflix today announced several new streaming features...",
    },
    {
        "title": "Disney+ Subscriber Growth",
        "url": "https://news.example.com/disney-growth",
        "snippet": "Disney+ reports strong subscriber growth in Q4...",
    },
    {
        "title": "Roku Expands Content Library",
        "url": "https://news.example.com/roku-content",
        "snippet": "Roku adds 100 new channels to free streaming service...",
    },
]


# =============================================================================
# Mock Article Data
# =============================================================================

def get_mock_articles(count: int = 3) -> list[dict]:
    """Generate mock article data for testing."""
    competitors = ["netflix", "disney", "youtube", "roku", "amazon"]
    sources = ["variety", "deadline", "techcrunch", "the_verge"]
    
    articles = []
    for i in range(count):
        comp = competitors[i % len(competitors)]
        source = sources[i % len(sources)]
        
        articles.append({
            "id": i + 1,
            "competitor_id": comp,
            "source_label": source,
            "title": f"{comp.title()} Streaming Update #{i + 1}",
            "url": f"https://example.com/{comp}-article-{i + 1}",
            "published_at": "2024-12-19T12:00:00Z",
            "raw_snippet": f"Test article about {comp} streaming service. "
                           f"This article covers recent developments and industry news.",
            "hash": f"mock_hash_{comp}_{i + 1}",
        })
    
    return articles


def get_mock_intel(count: int = 3) -> list[dict]:
    """Generate mock intel data for testing."""
    articles = get_mock_articles(count)
    categories = ["product", "content", "marketing", "ai_ads", "strategic"]
    
    intel_items = []
    for i, article in enumerate(articles):
        intel_items.append({
            "id": i + 1,
            "article_id": article["id"],
            "competitor_id": article["competitor_id"],
            "title": article["title"],
            "url": article["url"],
            "summary": f"Summary of {article['title']}: Important competitive development.",
            "category": categories[i % len(categories)],
            "relevance_score": 6.0 + (i % 4),
            "impact_score": 5.0 + (i % 5),
            "novelty_score": 0.7 + (i % 3) * 0.1,
            "entities": [article["competitor_id"].title(), "Streaming"],
            "annotations": [],
        })
    
    return intel_items
