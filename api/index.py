"""
Vercel Serverless Entry Point - Standalone Dashboard API

This is a self-contained Flask app for Vercel that only needs:
- Flask, SQLAlchemy, psycopg2 for the dashboard
- No LangChain, ChromaDB, OpenAI or heavy ML dependencies
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, jsonify, request, Response
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Database Setup (PostgreSQL/Supabase only for Vercel)
# ==============================================================================
def get_database_url():
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    return db_url

def get_engine():
    """Lazy engine creation for serverless."""
    db_url = get_database_url()
    if not db_url:
        return None
    return create_engine(
        db_url, 
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        connect_args={"sslmode": "require"}
    )

# ==============================================================================
# Simple Cache
# ==============================================================================
class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str):
        if key not in self._cache:
            return None
        if key in self._timestamps and time.time() > self._timestamps[key]:
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value, ttl_seconds: int = 300):
        self._cache[key] = value
        self._timestamps[key] = time.time() + ttl_seconds
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

cache = SimpleCache()

def cached(ttl_seconds: int = 300):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cache_key = f"{f.__name__}:{request.full_path}"
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                return Response(cached_response, mimetype='application/json', headers={'X-Cache': 'HIT'})
            result = f(*args, **kwargs)
            if hasattr(result, 'get_data'):
                cache.set(cache_key, result.get_data(as_text=True), ttl_seconds)
            return result
        return wrapper
    return decorator

# ==============================================================================
# Flask App
# ==============================================================================
template_dir = Path(__file__).parent.parent / "dashboard" / "templates"
static_dir = Path(__file__).parent.parent / "dashboard" / "static"

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

# ==============================================================================
# Page Routes
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html")

# ==============================================================================
# API Routes
# ==============================================================================
@app.route("/api/health")
def api_health():
    db_url = get_database_url()
    if not db_url:
        return jsonify({
            "status": "unhealthy", 
            "error": "DATABASE_URL not set",
            "hint": "Set DATABASE_URL environment variable in Vercel",
            "env_check": bool(os.environ.get('DATABASE_URL'))
        }), 500
    try:
        engine = get_engine()
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"status": "healthy", "database": "connected", "cache_enabled": True})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e), "db_url_length": len(db_url)}), 500


@app.route("/api/debug")
def api_debug():
    """Debug endpoint to check environment configuration."""
    db_url_raw = os.environ.get('DATABASE_URL', '')
    has_db = bool(db_url_raw)
    db_masked = f"{db_url_raw[:40]}...{db_url_raw[-15:]}" if len(db_url_raw) > 55 else f"[length={len(db_url_raw)}]"
    
    # Try to connect and get error if any
    db_test = "not tested"
    db_error = None
    if has_db:
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_test = "connected"
        except Exception as e:
            db_test = "failed"
            db_error = str(e)
    
    return jsonify({
        "database_url_set": has_db,
        "database_url_length": len(db_url_raw),
        "database_url_masked": db_masked if has_db else None,
        "database_url_starts_with": db_url_raw[:15] if db_url_raw else None,
        "db_test": db_test,
        "db_error": db_error,
        "environment": os.environ.get('VERCEL_ENV', 'local'),
        "all_env_keys": [k for k in os.environ.keys() if 'DATABASE' in k or 'VERCEL' in k or 'PYTHON' in k],
    })


@app.route("/api/stats")
@cached(ttl_seconds=120)
def api_stats():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "hint": "Set DATABASE_URL in Vercel"}), 500
    
    try:
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            runs = conn.execute(text("SELECT COUNT(*) FROM runs")).scalar() or 0
            articles = conn.execute(
                text("SELECT COUNT(*) FROM articles WHERE published_at >= :d"), 
                {"d": cutoff}
            ).scalar() or 0
            intel = conn.execute(text("""
                SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id 
                WHERE a.published_at >= :d
            """), {"d": cutoff}).scalar() or 0
            
            # Recent runs
            recent = conn.execute(text("""
                SELECT r.id, r.started_at, r.status,
                       (SELECT COUNT(*) FROM articles WHERE run_id = r.id) as articles_count,
                       (SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id WHERE a.run_id = r.id) as intel_count
                FROM runs r ORDER BY r.started_at DESC LIMIT 5
            """)).fetchall()
            
            return jsonify({
                "total_runs": runs,
                "total_articles": articles,
                "total_intel": intel,
                "competitors_tracked": 44,
                "sources_monitored": 178,
                "recent_runs": [
                    {
                        "id": r[0], 
                        "started_at": r[1].isoformat() if r[1] else None, 
                        "status": r[2], 
                        "articles_count": r[3] or 0, 
                        "intel_count": r[4] or 0
                    } 
                    for r in recent
                ]
            })
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/intel")
@cached(ttl_seconds=60)
def api_intel():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "intel": [], "total": 0}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        category = request.args.get('category', '')
        min_impact = request.args.get('min_impact', 0, type=float)
        min_relevance = request.args.get('min_relevance', 6, type=float)  # Default: only streaming-relevant
        competitor = request.args.get('competitor', '')
        sort_by = request.args.get('sort', 'date')  # 'date' or 'impact'
        
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            # Use DISTINCT ON to deduplicate by title, keeping highest impact
            # Filter by both impact AND relevance to streaming/CTV industry
            query = """
                SELECT DISTINCT ON (a.title) 
                       i.id, i.summary, i.category, i.impact_score, i.relevance_score,
                       i.related_urls_json, i.source_count, i.created_at,
                       a.id as article_id, a.title, a.url, a.published_at, a.source_label,
                       a.competitor_id
                FROM intel i
                JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff 
                  AND i.impact_score >= :min_impact
                  AND i.relevance_score >= :min_relevance
            """
            params = {"cutoff": cutoff, "min_impact": min_impact, "min_relevance": min_relevance}
            
            if category:
                query += " AND i.category = :category"
                params["category"] = category
            
            if competitor:
                query += " AND a.competitor_id = :competitor"
                params["competitor"] = competitor
            
            # DISTINCT ON requires ORDER BY to include the distinct column first
            query += " ORDER BY a.title, i.impact_score DESC"
            
            # Wrap in subquery to apply final sorting and limit
            if sort_by == 'impact':
                wrapper = f"SELECT * FROM ({query}) sub ORDER BY impact_score DESC, published_at DESC LIMIT :limit OFFSET :offset"
            else:
                wrapper = f"SELECT * FROM ({query}) sub ORDER BY published_at DESC LIMIT :limit OFFSET :offset"
            
            query = wrapper
            params["limit"] = limit
            params["offset"] = offset
            
            rows = conn.execute(text(query), params).fetchall()
            
            intel_list = []
            for r in rows:
                related_urls = []
                if r[5]:
                    try:
                        related_urls = json.loads(r[5]) if isinstance(r[5], str) else r[5]
                    except:
                        pass
                
                intel_list.append({
                    "id": r[0], 
                    "summary": r[1], 
                    "category": r[2], 
                    "impact_score": r[3], 
                    "relevance_score": r[4],
                    "related_urls": related_urls, 
                    "source_count": r[6] or 1,
                    "created_at": r[7].isoformat() if r[7] else None,
                    "article_id": r[8], 
                    "title": r[9], 
                    "url": r[10],
                    "published_at": r[11].isoformat() if r[11] else None,
                    "source_label": r[12],
                    "competitor_id": r[13],
                    "sources": [{"title": r[9], "url": r[10], "source": r[12]}]
                })
            
            # Get total count
            count_query = """
                SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff AND i.impact_score >= :min_impact
            """
            if category:
                count_query += " AND i.category = :category"
            
            total = conn.execute(text(count_query), params).scalar() or 0
            
            return jsonify({"intel": intel_list, "total": total, "limit": limit, "offset": offset})
            
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "type": type(e).__name__, 
            "intel": [], 
            "total": 0,
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/tubi/intel")
@cached(ttl_seconds=60)
def api_tubi_intel():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "intel": [], "total": 0}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            # Only show articles where the ORIGINAL article mentions Tubi
            # (title or raw article text - NOT the AI-generated summary)
            rows = conn.execute(text("""
                SELECT i.id, i.summary, i.category, i.impact_score, i.relevance_score,
                       i.related_urls_json, i.created_at, i.source_count,
                       a.id, a.title, a.url, a.published_at, a.source_label, a.competitor_id
                FROM intel i
                JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff
                  AND (LOWER(a.title) LIKE '%tubi%' 
                       OR LOWER(COALESCE(a.raw_snippet, '')) LIKE '%tubi%'
                       OR LOWER(COALESCE(a.summary, '')) LIKE '%tubi%')
                ORDER BY a.published_at DESC
                LIMIT :limit
            """), {"cutoff": cutoff, "limit": limit}).fetchall()
            
            intel_list = []
            for r in rows:
                related_urls = []
                if r[5]:
                    try:
                        related_urls = json.loads(r[5]) if isinstance(r[5], str) else r[5]
                    except:
                        pass
                
                intel_list.append({
                    "id": r[0], 
                    "summary": r[1], 
                    "category": r[2],
                    "impact_score": r[3], 
                    "relevance_score": r[4],
                    "related_urls": related_urls, 
                    "created_at": r[6].isoformat() if r[6] else None,
                    "source_count": r[7] or 1,
                    "article_id": r[8], 
                    "title": r[9], 
                    "url": r[10],
                    "published_at": r[11].isoformat() if r[11] else None,
                    "source_label": r[12],
                    "competitor_id": r[13],
                    "sources": [{"title": r[9], "url": r[10], "source": r[12]}]
                })
            
            return jsonify({"intel": intel_list, "total": len(intel_list)})
            
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "type": type(e).__name__, 
            "intel": [], 
            "total": 0,
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/tubi/stats")
@cached(ttl_seconds=120)
def api_tubi_stats():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "total_intel": 0, "high_impact": 0, "sources_tracked": 0}), 500
    
    try:
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            # Count only articles where ORIGINAL text mentions Tubi
            total = conn.execute(text("""
                SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff
                  AND (LOWER(a.title) LIKE '%tubi%' 
                       OR LOWER(COALESCE(a.raw_snippet, '')) LIKE '%tubi%'
                       OR LOWER(COALESCE(a.summary, '')) LIKE '%tubi%')
            """), {"cutoff": cutoff}).scalar() or 0
            
            high_impact = conn.execute(text("""
                SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff AND i.impact_score >= 8
                  AND (LOWER(a.title) LIKE '%tubi%' 
                       OR LOWER(COALESCE(a.raw_snippet, '')) LIKE '%tubi%'
                       OR LOWER(COALESCE(a.summary, '')) LIKE '%tubi%')
            """), {"cutoff": cutoff}).scalar() or 0
            
            return jsonify({
                "total_intel": total,
                "high_impact": high_impact,
                "sources_tracked": 12
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "type": type(e).__name__,
            "total_intel": 0, 
            "high_impact": 0, 
            "sources_tracked": 0
        }), 500


@app.route("/api/competitors")
@cached(ttl_seconds=300)
def api_competitors():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "competitors": []}), 500
    
    # Display names for competitor IDs
    display_names = {
        "netflix": "Netflix",
        "amazon": "Amazon Prime Video",
        "disney": "Disney+",
        "max": "Max (HBO)",
        "hulu": "Hulu",
        "peacock": "Peacock",
        "paramount": "Paramount+",
        "apple": "Apple TV+",
        "youtube": "YouTube/YouTube TV",
        "roku": "Roku Channel",
        "pluto": "Pluto TV",
        "fubo": "Fubo TV",
        "sling": "Sling TV",
        "directv": "DirecTV Stream",
        "fox": "Fox/Tubi",
        "xumo": "Xumo",
        "plex": "Plex",
        "vix": "ViX",
        "amc": "AMC+",
        "britbox": "BritBox",
        "samsung_tv_plus": "Samsung TV+",
        "lg_channels": "LG Channels",
        "vizio_watchfree": "Vizio WatchFree+",
        "filmrise": "FilmRise",
    }
    
    tier_map = {
        "netflix": "major", "amazon": "major", "disney": "major", "max": "major",
        "hulu": "major", "peacock": "major", "paramount": "major", "apple": "major",
        "youtube": "major", "roku": "fast", "pluto": "fast", "fubo": "live",
        "sling": "live", "directv": "live", "fox": "major", "xumo": "fast",
    }
    
    try:
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            # Query actual data from database
            rows = conn.execute(text("""
                SELECT a.competitor_id, 
                       COUNT(DISTINCT a.id) as article_count, 
                       COUNT(i.id) as intel_count
                FROM articles a
                LEFT JOIN intel i ON i.article_id = a.id
                WHERE a.published_at >= :cutoff 
                  AND a.competitor_id IS NOT NULL
                  AND a.competitor_id != 'industry'
                  AND a.competitor_id != 'tubi'
                GROUP BY a.competitor_id
                ORDER BY intel_count DESC
            """), {"cutoff": cutoff}).fetchall()
            
            competitors = []
            for r in rows:
                comp_id = r[0]
                competitors.append({
                    "id": comp_id,
                    "name": display_names.get(comp_id, comp_id.replace("_", " ").title()),
                    "tier": tier_map.get(comp_id, "other"),
                    "article_count": r[1],
                    "intel_count": r[2]
                })
        
        return jsonify({"competitors": competitors})
        
    except Exception as e:
        return jsonify({
            "competitors": [],
            "error": str(e)
        })


@app.route("/api/competitors/<competitor_id>/intel")
@cached(ttl_seconds=60)
def api_competitor_intel(competitor_id: str):
    if not get_database_url():
        return jsonify({"error": "Database not connected", "intel": [], "competitor_id": competitor_id}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        
        with get_engine().connect() as conn:
            cutoff = datetime(2025, 1, 1)
            
            rows = conn.execute(text("""
                SELECT i.id, i.summary, i.category, i.impact_score,
                       a.title, a.url, a.published_at, a.source_label
                FROM intel i
                JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff AND a.competitor_id = :cid
                ORDER BY a.published_at DESC
                LIMIT :limit
            """), {"cutoff": cutoff, "cid": competitor_id, "limit": limit}).fetchall()
            
            intel_list = [{
                "id": r[0], 
                "summary": r[1], 
                "category": r[2], 
                "impact_score": r[3],
                "title": r[4], 
                "url": r[5], 
                "published_at": r[6].isoformat() if r[6] else None,
                "source_label": r[7]
            } for r in rows]
            
            return jsonify({"intel": intel_list, "competitor_id": competitor_id})
            
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "intel": [], 
            "competitor_id": competitor_id
        }), 500


@app.route("/api/last-updated")
@cached(ttl_seconds=60)
def api_last_updated():
    if not get_database_url():
        return jsonify({"error": "Database not connected", "last_article": None, "last_intel": None}), 500
    
    try:
        with get_engine().connect() as conn:
            last_article = conn.execute(text("SELECT MAX(created_at) FROM articles")).scalar()
            last_intel = conn.execute(text("SELECT MAX(created_at) FROM intel")).scalar()
            return jsonify({
                "last_article": last_article.isoformat() if last_article else None,
                "last_intel": last_intel.isoformat() if last_intel else None
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "last_article": None, 
            "last_intel": None
        }), 500


@app.route("/api/reports")
def api_reports():
    return jsonify({"reports": [], "total": 0})


@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    cache.clear()
    return jsonify({"status": "ok"})


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500


# For local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
