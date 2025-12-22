"""
Tubi Radar Dashboard - Flask Application

Premium Tubi-inspired competitive intelligence dashboard.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from radar.database import get_session_factory, init_database
from radar.models import Run, Article, Intel, Annotation, Report
from radar.config import load_config

app = Flask(__name__)

# Initialize database
db_path = project_root / "data" / "radar.db"
init_database(db_path)
Session = get_session_factory(db_path)

# Load config for competitor info
try:
    config = load_config(project_root / "config" / "radar.yaml")
except Exception:
    config = None


# ==============================================================================
# Page Routes
# ==============================================================================

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


# ==============================================================================
# API Routes - Stats
# ==============================================================================

@app.route("/api/stats")
def api_stats():
    """Get overall dashboard statistics."""
    session = Session()
    try:
        total_runs = session.query(Run).count()
        total_articles = session.query(Article).count()
        total_intel = session.query(Intel).count()
        
        # Get recent runs with details
        recent_runs = session.query(Run).order_by(Run.id.desc()).limit(10).all()
        runs_data = []
        for run in recent_runs:
            articles_count = session.query(Article).filter(Article.run_id == run.id).count()
            # Intel is linked through articles
            intel_count = session.query(Intel).join(Article).filter(Article.run_id == run.id).count()
            
            # Check for report path on Run model
            report_path = getattr(run, 'report_path', None)
            if not report_path:
                # Try to find from Report table
                report = session.query(Report).filter(Report.run_id == run.id).first()
                report_path = report.path if report else None
            
            runs_data.append({
                "id": run.id,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "status": run.status,
                "articles_count": articles_count,
                "intel_count": intel_count,
                "report_path": report_path
            })
        
        # Count competitors and sources from config
        competitors_tracked = 0
        sources_monitored = 0
        if config:
            competitors_tracked = len(config.competitors)
            for comp in config.competitors:
                sources_monitored += len(comp.feeds)
            sources_monitored += len(config.industry_feeds)
        
        return jsonify({
            "total_runs": total_runs,
            "total_articles": total_articles,
            "total_intel": total_intel,
            "competitors_tracked": competitors_tracked,
            "sources_monitored": sources_monitored,
            "recent_runs": runs_data
        })
    finally:
        session.close()


# ==============================================================================
# API Routes - Intel
# ==============================================================================

@app.route("/api/intel")
def api_intel():
    """Get intel items with optional filters."""
    session = Session()
    try:
        limit = request.args.get("limit", 50, type=int)
        category = request.args.get("category", "")
        competitor = request.args.get("competitor", "")
        min_impact = request.args.get("min_impact", 0, type=float)
        
        query = session.query(Intel, Article).join(Article).order_by(Intel.impact_score.desc())
        
        if category:
            query = query.filter(Intel.category == category)
        if competitor:
            query = query.filter(Article.competitor_id == competitor)
        if min_impact > 0:
            query = query.filter(Intel.impact_score >= min_impact)
        
        results = query.limit(limit).all()
        
        intel_list = []
        for intel_item, article in results:
            # Parse related URLs
            related_urls = []
            if hasattr(intel_item, 'related_urls_json') and intel_item.related_urls_json:
                try:
                    import json
                    related_urls = json.loads(intel_item.related_urls_json)
                except:
                    pass
            
            intel_list.append({
                "id": intel_item.id,
                "article_id": intel_item.article_id,
                "competitor_id": article.competitor_id,
                "title": article.title,
                "summary": intel_item.summary,
                "category": intel_item.category,
                "impact_score": intel_item.impact_score,
                "relevance_score": intel_item.relevance_score,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "novelty_score": intel_item.novelty_score,
                "source_count": getattr(intel_item, 'source_count', 1) or 1,
                "related_urls": related_urls,
            })
        
        return jsonify({"intel": intel_list, "total": len(intel_list)})
    finally:
        session.close()


# ==============================================================================
# API Routes - Tubi (Our Company)
# ==============================================================================

@app.route("/api/tubi/articles")
def api_tubi_articles():
    """Get articles specifically about Tubi."""
    session = Session()
    try:
        import json
        limit = request.args.get("limit", 100, type=int)
        
        # Get articles that mention Tubi in title or are from Tubi sources
        query = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Article.competitor_id == 'tubi') |
            (Article.source_label.ilike('%tubi%'))
        ).order_by(Article.published_at.desc())
        
        articles = query.limit(limit).all()
        
        result = []
        for article in articles:
            # Check if there's intel for this article
            intel = session.query(Intel).filter(Intel.article_id == article.id).first()
            
            result.append({
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "source": article.source_label,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "has_intel": intel is not None,
                "summary": intel.summary if intel else None,
                "category": intel.category if intel else None,
                "impact_score": intel.impact_score if intel else None,
            })
        
        return jsonify({"articles": result, "total": len(result)})
    finally:
        session.close()


@app.route("/api/tubi/intel")
def api_tubi_intel():
    """Get intel items that mention Tubi."""
    session = Session()
    try:
        import json
        limit = request.args.get("limit", 50, type=int)
        
        # Get intel where the article mentions Tubi
        query = session.query(Intel, Article).join(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Intel.summary.ilike('%tubi%')) |
            (Article.competitor_id == 'tubi')
        ).order_by(Intel.impact_score.desc())
        
        results = query.limit(limit).all()
        
        intel_list = []
        for intel_item, article in results:
            related_urls = []
            if hasattr(intel_item, 'related_urls_json') and intel_item.related_urls_json:
                try:
                    related_urls = json.loads(intel_item.related_urls_json)
                except:
                    pass
            
            intel_list.append({
                "id": intel_item.id,
                "article_id": intel_item.article_id,
                "competitor_id": article.competitor_id,
                "title": article.title,
                "summary": intel_item.summary,
                "category": intel_item.category,
                "impact_score": intel_item.impact_score,
                "relevance_score": intel_item.relevance_score,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "source_count": getattr(intel_item, 'source_count', 1) or 1,
                "related_urls": related_urls,
            })
        
        return jsonify({"intel": intel_list, "total": len(intel_list)})
    finally:
        session.close()


@app.route("/api/tubi/stats")
def api_tubi_stats():
    """Get Tubi-specific statistics."""
    session = Session()
    try:
        # Count articles mentioning Tubi
        tubi_articles = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Article.competitor_id == 'tubi')
        ).count()
        
        # Count intel mentioning Tubi
        tubi_intel = session.query(Intel).join(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Intel.summary.ilike('%tubi%'))
        ).count()
        
        # Get category breakdown
        category_query = session.query(Intel.category, session.query(Intel).count()).join(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Intel.summary.ilike('%tubi%'))
        ).group_by(Intel.category)
        
        # Get recent Tubi mentions count by day (last 7 days)
        from datetime import datetime, timedelta
        week_ago = datetime.now() - timedelta(days=7)
        recent_count = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) | 
            (Article.competitor_id == 'tubi'),
            Article.published_at >= week_ago
        ).count()
        
        return jsonify({
            "total_articles": tubi_articles,
            "total_intel": tubi_intel,
            "recent_mentions": recent_count,
        })
    finally:
        session.close()


# ==============================================================================
# API Routes - Competitors
# ==============================================================================

@app.route("/api/competitors")
def api_competitors():
    """Get competitor information from config."""
    if not config:
        return jsonify({"competitors": []})
    
    competitors = []
    for comp in config.competitors:
        competitors.append({
            "id": comp.id,
            "name": comp.name,
            "category": comp.category or "streaming",
            "feeds_count": len(comp.feeds),
            "search_queries": len(comp.search_queries) if comp.search_queries else 0
        })
    
    return jsonify({"competitors": competitors})


# ==============================================================================
# API Routes - Reports
# ==============================================================================

@app.route("/api/reports")
def api_reports():
    """List all available reports."""
    session = Session()
    try:
        result = []
        
        # First check Report table
        reports = session.query(Report).order_by(Report.id.desc()).all()
        for report in reports:
            if Path(report.path).exists():
                result.append({
                    "id": report.id,
                    "run_id": report.run_id,
                    "created_at": report.created_at.isoformat() if report.created_at else None,
                    "path": report.path
                })
        
        # Also check Run.report_path for runs not in Report table
        report_run_ids = {r.run_id for r in reports}
        runs = session.query(Run).order_by(Run.id.desc()).all()
        for run in runs:
            if run.id in report_run_ids:
                continue
            report_path = getattr(run, 'report_path', None)
            if report_path and Path(report_path).exists():
                result.append({
                    "id": run.id,
                    "run_id": run.id,
                    "created_at": run.finished_at.isoformat() if run.finished_at else None,
                    "path": report_path
                })
        
        return jsonify({"reports": result})
    finally:
        session.close()


@app.route("/api/reports/<int:run_id>")
def api_report_content(run_id: int):
    """Get the content of a specific report."""
    session = Session()
    try:
        # Try to find report by run_id
        report = session.query(Report).filter(Report.run_id == run_id).first()
        
        if not report:
            # Try run's report_path attribute
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                report_path_attr = getattr(run, 'report_path', None)
                if report_path_attr:
                    report_path = Path(report_path_attr)
                    if not report_path.exists():
                        report_path = project_root / report_path_attr
                    if report_path.exists():
                        content = report_path.read_text(encoding="utf-8")
                        return jsonify({
                            "run_id": run_id,
                            "content": content,
                            "path": str(report_path)
                        })
            return jsonify({"error": "Report not found"}), 404
        
        report_path = Path(report.path)
        if not report_path.exists():
            # Try relative to project root
            report_path = project_root / report.path
        
        if not report_path.exists():
            return jsonify({"error": "Report file not found"}), 404
        
        content = report_path.read_text(encoding="utf-8")
        return jsonify({
            "run_id": run_id,
            "content": content,
            "path": str(report_path)
        })
    finally:
        session.close()


# ==============================================================================
# API Routes - Annotations
# ==============================================================================

@app.route("/api/annotations")
def api_annotations():
    """Get domain annotations."""
    session = Session()
    try:
        limit = request.args.get("limit", 50, type=int)
        agent = request.args.get("agent", "")
        
        query = session.query(Annotation).order_by(Annotation.id.desc())
        if agent:
            query = query.filter(Annotation.agent_role == agent)
        
        annotations = query.limit(limit).all()
        
        result = []
        for ann in annotations:
            result.append({
                "id": ann.id,
                "intel_id": ann.intel_id,
                "agent_role": ann.agent_role,
                "so_what": ann.so_what,
                "risk_opportunity": ann.risk_opportunity,
                "priority": ann.priority,
                "suggested_action": ann.suggested_action
            })
        
        return jsonify({"annotations": result, "total": len(result)})
    finally:
        session.close()


# ==============================================================================
# Error Handlers
# ==============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
