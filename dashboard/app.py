"""
Competitor Monitor Dashboard - Flask Application

Premium competitive intelligence dashboard.
Supports both SQLite (local) and PostgreSQL (Supabase production).
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template, jsonify, request, Response


# ==============================================================================
# Simple In-Memory Cache
# ==============================================================================
class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str):
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        
        # Check if expired
        if key in self._timestamps:
            if time.time() > self._timestamps[key]:
                del self._cache[key]
                del self._timestamps[key]
                return None
        
        return self._cache[key]
    
    def set(self, key: str, value, ttl_seconds: int = 300):
        """Set cache with TTL (default 5 minutes)."""
        self._cache[key] = value
        self._timestamps[key] = time.time() + ttl_seconds
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._timestamps.clear()


# Global cache instance
cache = SimpleCache()


def cached(ttl_seconds: int = 300):
    """Decorator to cache API responses."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and request args
            cache_key = f"{f.__name__}:{request.full_path}"
            
            # Check cache
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                return Response(
                    cached_response,
                    mimetype='application/json',
                    headers={'X-Cache': 'HIT'}
                )
            
            # Call function
            result = f(*args, **kwargs)
            
            # Cache the response
            if isinstance(result, Response):
                cache.set(cache_key, result.get_data(as_text=True), ttl_seconds)
            else:
                response_data = json.dumps(result.json) if hasattr(result, 'json') else str(result)
                cache.set(cache_key, response_data, ttl_seconds)
            
            return result
        return wrapper
    return decorator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from radar.database import get_session_factory, init_database, is_postgres
from radar.models import Run, Article, Intel, Annotation, Report
from radar.config import load_config

app = Flask(__name__)

# Initialize database (works with both SQLite and PostgreSQL)
# If DATABASE_URL is set, uses PostgreSQL (Supabase)
# Otherwise, uses local SQLite
if is_postgres():
    print("Using PostgreSQL (Supabase)")
    init_database()  # No path needed for PostgreSQL
    Session = get_session_factory()
else:
    print("Using SQLite (local)")
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
@cached(ttl_seconds=120)  # Cache for 2 minutes
def api_stats():
    """Get overall dashboard statistics (2025+ only)."""
    session = Session()
    try:
        from datetime import datetime
        cutoff_date = datetime(2025, 1, 1)
        
        total_runs = session.query(Run).count()
        total_articles = session.query(Article).filter(Article.published_at >= cutoff_date).count()
        total_intel = session.query(Intel).join(Article).filter(Article.published_at >= cutoff_date).count()
        
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
@cached(ttl_seconds=60)  # Cache for 1 minute
def api_intel():
    """Get intel items with optional filters (2025+ only, ordered by recency)."""
    session = Session()
    try:
        from datetime import datetime
        limit = request.args.get("limit", 50, type=int)
        category = request.args.get("category", "")
        competitor = request.args.get("competitor", "")
        min_impact = request.args.get("min_impact", 0, type=float)
        sort_by = request.args.get("sort", "recent")  # 'recent' or 'impact'
        
        # Only show 2025+ articles
        cutoff_date = datetime(2025, 1, 1)
        
        query = session.query(Intel, Article).join(Article).filter(
            Article.published_at >= cutoff_date
        )
        
        # Sort by recency (default) or impact
        if sort_by == "impact":
            query = query.order_by(Intel.impact_score.desc(), Article.published_at.desc())
        else:
            query = query.order_by(Article.published_at.desc(), Intel.impact_score.desc())
        
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
@cached(ttl_seconds=120)  # Cache for 2 minutes
def api_tubi_articles():
    """Get articles specifically about Tubi (2025+ only, ordered by recency)."""
    session = Session()
    try:
        import json
        from datetime import datetime
        limit = request.args.get("limit", 100, type=int)
        
        # Only 2025+
        cutoff_date = datetime(2025, 1, 1)
        
        # STRICT: Only articles where Tubi is in the title OR article content
        # Ordered by recency (most recent first)
        query = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) |
            (Article.raw_snippet.ilike('%tubi%')),
            Article.published_at >= cutoff_date
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
@cached(ttl_seconds=60)  # Cache for 1 minute
def api_tubi_intel():
    """Get intel items ABOUT Tubi (2025+ only, ordered by recency)."""
    session = Session()
    try:
        import json
        from datetime import datetime
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "recent")  # 'recent' or 'impact'
        
        # Only 2025+
        cutoff_date = datetime(2025, 1, 1)
        
        # STRICT: Only intel where the ARTICLE (title or snippet) mentions Tubi
        query = session.query(Intel, Article).join(Article).filter(
            (Article.title.ilike('%tubi%')) |
            (Article.raw_snippet.ilike('%tubi%')),
            Article.published_at >= cutoff_date
        )
        
        # Sort by recency (default) or impact
        if sort_by == "impact":
            query = query.order_by(Intel.impact_score.desc(), Article.published_at.desc())
        else:
            query = query.order_by(Article.published_at.desc(), Intel.impact_score.desc())
        
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
@cached(ttl_seconds=120)  # Cache for 2 minutes
def api_tubi_stats():
    """Get Tubi-specific statistics (2025+ only, article must mention Tubi)."""
    session = Session()
    try:
        from datetime import datetime, timedelta
        
        # Only 2025+
        cutoff_date = datetime(2025, 1, 1)
        
        # Count articles where Tubi is in title or content (2025+)
        tubi_articles = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) |
            (Article.raw_snippet.ilike('%tubi%')),
            Article.published_at >= cutoff_date
        ).count()
        
        # Count intel where the SOURCE ARTICLE mentions Tubi (2025+)
        tubi_intel = session.query(Intel).join(Article).filter(
            (Article.title.ilike('%tubi%')) |
            (Article.raw_snippet.ilike('%tubi%')),
            Article.published_at >= cutoff_date
        ).count()
        
        # Get recent (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_count = session.query(Article).filter(
            (Article.title.ilike('%tubi%')) |
            (Article.raw_snippet.ilike('%tubi%')),
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
@cached(ttl_seconds=300)  # Cache for 5 minutes
def api_competitors():
    """Get competitor information from config with intel counts."""
    session = Session()
    try:
        from datetime import datetime
        cutoff_date = datetime(2025, 1, 1)
        
        if not config:
            return jsonify({"competitors": []})
        
        competitors = []
        for comp in config.competitors:
            # Count intel for this competitor
            intel_count = session.query(Intel).join(Article).filter(
                Article.competitor_id == comp.id,
                Article.published_at >= cutoff_date
            ).count()
            
            # Count articles
            article_count = session.query(Article).filter(
                Article.competitor_id == comp.id,
                Article.published_at >= cutoff_date
            ).count()
            
            competitors.append({
                "id": comp.id,
                "name": comp.name,
                "category": comp.category or "streaming",
                "feeds_count": len(comp.feeds),
                "search_queries": len(comp.search_queries) if comp.search_queries else 0,
                "intel_count": intel_count,
                "article_count": article_count,
            })
        
        # Sort by intel count
        competitors.sort(key=lambda x: x["intel_count"], reverse=True)
        
        return jsonify({"competitors": competitors})
    finally:
        session.close()


@app.route("/api/competitors/<competitor_id>/intel")
@cached(ttl_seconds=60)  # Cache for 1 minute
def api_competitor_intel(competitor_id: str):
    """Get all intel for a specific competitor (ordered by recency)."""
    session = Session()
    try:
        import json
        from datetime import datetime
        cutoff_date = datetime(2025, 1, 1)
        
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "recent")  # 'recent' or 'impact'
        
        query = session.query(Intel, Article).join(Article).filter(
            Article.competitor_id == competitor_id,
            Article.published_at >= cutoff_date
        )
        
        # Sort by recency (default) or impact
        if sort_by == "impact":
            query = query.order_by(Intel.impact_score.desc(), Article.published_at.desc())
        else:
            query = query.order_by(Article.published_at.desc(), Intel.impact_score.desc())
        
        results = query.limit(limit).all()
        
        intel_list = []
        for intel_item, article in results:
            related_urls = []
            if intel_item.related_urls_json:
                try:
                    related_urls = json.loads(intel_item.related_urls_json)
                except:
                    pass
            
            intel_list.append({
                "id": intel_item.id,
                "article_id": intel_item.article_id,
                "title": article.title,
                "summary": intel_item.summary,
                "category": intel_item.category,
                "impact_score": intel_item.impact_score,
                "relevance_score": intel_item.relevance_score,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "source_count": intel_item.source_count or 1,
                "related_urls": related_urls,
            })
        
        return jsonify({"competitor_id": competitor_id, "intel": intel_list, "total": len(intel_list)})
    finally:
        session.close()


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
# Cache Management
# ==============================================================================

@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    """Clear all cached responses."""
    cache.clear()
    return jsonify({"status": "ok", "message": "Cache cleared"})


@app.route("/api/health")
def api_health():
    """Health check endpoint."""
    session = Session()
    try:
        # Quick DB check
        from sqlalchemy import text
        session.execute(text("SELECT 1"))
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "cache_enabled": True,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
    finally:
        session.close()


@app.route("/api/last-updated")
@cached(ttl_seconds=60)
def api_last_updated():
    """Get last data update timestamp."""
    session = Session()
    try:
        from sqlalchemy import func
        
        # Get most recent article
        last_article = session.query(func.max(Article.created_at)).scalar()
        last_intel = session.query(func.max(Intel.created_at)).scalar()
        
        return jsonify({
            "last_article": last_article.isoformat() if last_article else None,
            "last_intel": last_intel.isoformat() if last_intel else None,
            "checked_at": datetime.now().isoformat()
        })
    finally:
        session.close()


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
