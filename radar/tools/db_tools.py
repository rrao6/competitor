"""
Database read/write tools for agents.

Provides LangChain-compatible tools for storing and retrieving data
from the SQLite blackboard.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy import select, and_
from langchain_core.tools import tool

from radar.database import get_session
from radar.models import Run, Article, Intel, Annotation, Report
from radar.tools.rss import ArticleCandidate
from radar.schemas import ArticleClassification, DomainAnnotation


# =============================================================================
# Run Management
# =============================================================================

def create_run() -> int:
    """Create a new run and return its ID."""
    with get_session() as session:
        run = Run(status="running")
        session.add(run)
        session.flush()
        return run.id


def complete_run(run_id: int, status: str = "success", notes: Optional[str] = None) -> None:
    """Mark a run as complete."""
    with get_session() as session:
        run = session.get(Run, run_id)
        if run:
            run.status = status
            run.finished_at = datetime.utcnow()
            run.notes = notes


def get_run(run_id: int) -> Optional[dict]:
    """Get run details."""
    with get_session() as session:
        run = session.get(Run, run_id)
        if run:
            return {
                "id": run.id,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "status": run.status,
                "notes": run.notes,
            }
        return None


# =============================================================================
# Article Management
# =============================================================================

@tool
def store_articles(run_id: int, items: list[dict]) -> int:
    """
    Store article candidates in the database.
    
    Skips articles with duplicate hashes or URLs.
    
    Args:
        run_id: The current run ID
        items: List of article candidate dictionaries
    
    Returns:
        Number of new articles stored
    """
    stored_count = 0
    
    with get_session() as session:
        # Get existing hashes AND URLs for deduplication
        existing_hashes = set(
            row[0] for row in session.execute(
                select(Article.hash)
            ).all() if row[0]
        )
        existing_urls = set(
            row[0] for row in session.execute(
                select(Article.url)
            ).all() if row[0]
        )
        
        for item in items:
            article_hash = item.get("hash", "")
            article_url = item.get("url", "")
            
            # Skip if we already have this article (by hash OR URL)
            if article_hash and article_hash in existing_hashes:
                continue
            if article_url and article_url in existing_urls:
                continue
            
            # Parse published_at
            published_at = item.get("published_at")
            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    published_at = None
            
            article = Article(
                run_id=run_id,
                competitor_id=item.get("competitor_id", "unknown"),
                source_label=item.get("source_label", "unknown"),
                title=item.get("title", ""),
                url=item.get("url", ""),
                published_at=published_at,
                raw_snippet=item.get("raw_snippet", ""),
                hash=article_hash,
            )
            session.add(article)
            existing_hashes.add(article_hash)
            existing_urls.add(article_url)
            stored_count += 1
    
    return stored_count


def store_articles_batch(run_id: int, candidates: list[ArticleCandidate]) -> int:
    """
    Store article candidates directly from ArticleCandidate objects.
    
    Args:
        run_id: The current run ID
        candidates: List of ArticleCandidate objects
    
    Returns:
        Number of new articles stored
    """
    items = []
    for c in candidates:
        items.append({
            "competitor_id": c.competitor_id,
            "source_label": c.source_label,
            "title": c.title,
            "url": c.url,
            "published_at": c.published_at.isoformat() if c.published_at else None,
            "raw_snippet": c.raw_snippet,
            "hash": c.hash,
        })
    return store_articles.invoke({"run_id": run_id, "items": items})


@tool
def get_unprocessed_articles(run_id: int, limit: int = 50) -> list[dict]:
    """
    Get articles from this run that haven't been processed into intel yet.
    
    Args:
        run_id: The current run ID
        limit: Maximum number of articles to return
    
    Returns:
        List of article dictionaries
    """
    with get_session() as session:
        # Find articles without corresponding intel
        stmt = (
            select(Article)
            .outerjoin(Intel, Article.id == Intel.article_id)
            .where(
                and_(
                    Article.run_id == run_id,
                    Intel.id.is_(None)
                )
            )
            .limit(limit)
        )
        
        articles = session.execute(stmt).scalars().all()
        
        return [
            {
                "id": a.id,
                "competitor_id": a.competitor_id,
                "source_label": a.source_label,
                "title": a.title,
                "url": a.url,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "raw_snippet": a.raw_snippet,
            }
            for a in articles
        ]


# =============================================================================
# Intel Management
# =============================================================================

@tool
def store_intel(records: list[dict]) -> int:
    """
    Store intel records from the Understanding Agent.
    
    Args:
        records: List of intel dictionaries with article_id, summary, category, scores, entities
    
    Returns:
        Number of intel records stored
    """
    stored_count = 0
    
    with get_session() as session:
        for record in records:
            intel = Intel(
                article_id=record["article_id"],
                summary=record["summary"],
                category=record["category"],
                relevance_score=record["relevance_score"],
                impact_score=record["impact_score"],
                entities_json=json.dumps(record.get("entities", [])),
                llm_metadata=record.get("llm_metadata"),
            )
            session.add(intel)
            stored_count += 1
    
    return stored_count


def store_intel_from_classifications(classifications: list[ArticleClassification]) -> int:
    """
    Store intel from ArticleClassification objects.
    
    Args:
        classifications: List of ArticleClassification objects
    
    Returns:
        Number of intel records stored
    """
    records = []
    for c in classifications:
        records.append({
            "article_id": c.article_id,
            "summary": c.summary,
            "category": c.category,
            "relevance_score": c.relevance_score,
            "impact_score": c.impact_score,
            "entities": c.entities,
        })
    return store_intel.invoke({"records": records})


@tool
def get_recent_intel_for_dedup(window_days: int = 30) -> list[dict]:
    """
    Get recent intel for deduplication checking.
    
    Args:
        window_days: Number of days to look back
    
    Returns:
        List of intel dictionaries
    """
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    
    with get_session() as session:
        stmt = (
            select(Intel)
            .join(Article, Intel.article_id == Article.id)
            .where(Intel.created_at >= cutoff)
            .order_by(Intel.created_at.desc())
        )
        
        intels = session.execute(stmt).scalars().all()
        
        return [
            {
                "id": i.id,
                "article_id": i.article_id,
                "summary": i.summary,
                "category": i.category,
                "relevance_score": i.relevance_score,
                "impact_score": i.impact_score,
                "novelty_score": i.novelty_score,
                "is_duplicate_of": i.is_duplicate_of,
                "url": i.article.url if i.article else None,
            }
            for i in intels
        ]


@tool
def store_novelty_scores(updates: list[dict]) -> int:
    """
    Update novelty scores and duplicate flags for intel items.
    
    Args:
        updates: List of dicts with intel_id, novelty_score, is_duplicate_of
    
    Returns:
        Number of records updated
    """
    updated_count = 0
    
    with get_session() as session:
        for update in updates:
            intel = session.get(Intel, update["intel_id"])
            if intel:
                intel.novelty_score = update.get("novelty_score")
                if update.get("is_duplicate_of"):
                    intel.is_duplicate_of = update["is_duplicate_of"]
                updated_count += 1
    
    return updated_count


@tool
def get_intel_for_domain(
    run_id: int,
    category_filter: list[str],
    min_relevance: float = 4.0,
    min_impact: float = 4.0,
    min_novelty: float = 0.0,
) -> list[dict]:
    """
    Get filtered intel for domain agents.
    
    Args:
        run_id: The current run ID
        category_filter: List of categories to include
        min_relevance: Minimum relevance score
        min_impact: Minimum impact score
        min_novelty: Minimum novelty score
    
    Returns:
        List of intel dictionaries with article info
    """
    with get_session() as session:
        stmt = (
            select(Intel)
            .join(Article, Intel.article_id == Article.id)
            .where(
                and_(
                    Article.run_id == run_id,
                    Intel.category.in_(category_filter),
                    Intel.relevance_score >= min_relevance,
                    Intel.impact_score >= min_impact,
                    Intel.is_duplicate_of.is_(None),
                )
            )
            .order_by(Intel.impact_score.desc())
        )
        
        # Optional novelty filter
        if min_novelty > 0:
            stmt = stmt.where(Intel.novelty_score >= min_novelty)
        
        intels = session.execute(stmt).scalars().all()
        
        return [
            {
                "id": i.id,
                "article_id": i.article_id,
                "competitor_id": i.article.competitor_id if i.article else "unknown",
                "summary": i.summary,
                "category": i.category,
                "relevance_score": i.relevance_score,
                "impact_score": i.impact_score,
                "novelty_score": i.novelty_score,
                "entities": json.loads(i.entities_json) if i.entities_json else [],
            }
            for i in intels
        ]


def get_all_intel_for_run(run_id: int, min_relevance: float = 4.0, min_impact: float = 4.0) -> list[dict]:
    """
    Get all qualifying intel for a run (for report generation).
    
    Args:
        run_id: The current run ID
        min_relevance: Minimum relevance score
        min_impact: Minimum impact score
    
    Returns:
        List of intel dictionaries with annotations
    """
    with get_session() as session:
        stmt = (
            select(Intel)
            .join(Article, Intel.article_id == Article.id)
            .where(
                and_(
                    Article.run_id == run_id,
                    Intel.relevance_score >= min_relevance,
                    Intel.impact_score >= min_impact,
                    Intel.is_duplicate_of.is_(None),
                    Intel.category != "noise",
                )
            )
            .order_by(Intel.impact_score.desc())
        )
        
        intels = session.execute(stmt).scalars().all()
        
        results = []
        for i in intels:
            # Get annotations for this intel
            annotations = []
            for a in i.annotations:
                annotations.append({
                    "agent_role": a.agent_role,
                    "so_what": a.so_what,
                    "risk_opportunity": a.risk_opportunity,
                    "priority": a.priority,
                    "suggested_action": a.suggested_action,
                })
            
            results.append({
                "id": i.id,
                "article_id": i.article_id,
                "competitor_id": i.article.competitor_id if i.article else "unknown",
                "title": i.article.title if i.article else "",
                "url": i.article.url if i.article else "",
                "summary": i.summary,
                "category": i.category,
                "relevance_score": i.relevance_score,
                "impact_score": i.impact_score,
                "novelty_score": i.novelty_score,
                "entities": json.loads(i.entities_json) if i.entities_json else [],
                "annotations": annotations,
            })
        
        return results


# =============================================================================
# Annotation Management
# =============================================================================

@tool
def store_annotations(records: list[dict]) -> int:
    """
    Store annotations from domain agents.
    
    Args:
        records: List of annotation dictionaries
    
    Returns:
        Number of annotations stored
    """
    stored_count = 0
    
    with get_session() as session:
        for record in records:
            annotation = Annotation(
                intel_id=record["intel_id"],
                agent_role=record["agent_role"],
                so_what=record["so_what"],
                risk_opportunity=record["risk_opportunity"],
                priority=record["priority"],
                suggested_action=record.get("suggested_action"),
            )
            session.add(annotation)
            stored_count += 1
    
    return stored_count


def store_annotations_from_batch(
    annotations: list[DomainAnnotation],
    agent_role: str,
) -> int:
    """
    Store annotations from DomainAnnotation objects.
    
    Args:
        annotations: List of DomainAnnotation objects
        agent_role: The role of the agent creating these annotations
    
    Returns:
        Number of annotations stored
    """
    records = []
    for a in annotations:
        records.append({
            "intel_id": a.intel_id,
            "agent_role": agent_role,
            "so_what": a.so_what,
            "risk_opportunity": a.risk_or_opportunity,
            "priority": a.priority,
            "suggested_action": a.suggested_action,
        })
    return store_annotations.invoke({"records": records})


# =============================================================================
# Report Management
# =============================================================================

@tool
def create_report_file(run_id: int, content_markdown: str) -> str:
    """
    Create a markdown report file and store its metadata.
    
    Args:
        run_id: The current run ID
        content_markdown: The markdown content of the report
    
    Returns:
        Path to the created report file
    """
    # Generate filename with date
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"radar-{date_str}-run{run_id}.md"
    
    # Determine reports directory
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / filename
    
    # Write the file
    with open(report_path, "w") as f:
        f.write(content_markdown)
    
    # Store metadata in database
    with get_session() as session:
        # Extract summary excerpt (first section after title)
        lines = content_markdown.split("\n")
        excerpt_lines = []
        in_excerpt = False
        for line in lines:
            if line.startswith("## "):
                if in_excerpt:
                    break
                in_excerpt = True
            elif in_excerpt:
                excerpt_lines.append(line)
                if len(excerpt_lines) >= 5:
                    break
        
        excerpt = "\n".join(excerpt_lines).strip()[:500]
        
        report = Report(
            run_id=run_id,
            path=str(report_path),
            summary_excerpt=excerpt,
        )
        session.add(report)
    
    return str(report_path)

