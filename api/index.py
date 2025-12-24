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
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, jsonify, request, Response
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
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

db_url = get_database_url()
engine = create_engine(db_url, pool_pre_ping=True) if db_url else None
Session = sessionmaker(bind=engine) if engine else None

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
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"status": "healthy", "database": "connected", "cache_enabled": True})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/api/stats")
@cached(ttl_seconds=120)
def api_stats():
    with engine.connect() as conn:
        cutoff = datetime(2025, 1, 1)
        
        runs = conn.execute(text("SELECT COUNT(*) FROM runs")).scalar()
        articles = conn.execute(text("SELECT COUNT(*) FROM articles WHERE published_at >= :d"), {"d": cutoff}).scalar()
        intel = conn.execute(text("""
            SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id 
            WHERE a.published_at >= :d
        """), {"d": cutoff}).scalar()
        
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
            "recent_runs": [{"id": r[0], "started_at": r[1].isoformat() if r[1] else None, 
                           "status": r[2], "articles_count": r[3], "intel_count": r[4]} for r in recent]
        })

@app.route("/api/intel")
@cached(ttl_seconds=60)
def api_intel():
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    category = request.args.get('category', '')
    min_impact = request.args.get('min_impact', 0, type=float)
    
    with engine.connect() as conn:
        cutoff = datetime(2025, 1, 1)
        
        query = """
            SELECT i.id, i.summary, i.category, i.impact_score, i.relevance_score,
                   i.entities_json, i.source_count, i.created_at,
                   a.id as article_id, a.title, a.url, a.published_at, a.source_label
            FROM intel i
            JOIN articles a ON i.article_id = a.id
            WHERE a.published_at >= :cutoff AND i.impact_score >= :min_impact
        """
        params = {"cutoff": cutoff, "min_impact": min_impact}
        
        if category:
            query += " AND i.category = :category"
            params["category"] = category
        
        query += " ORDER BY a.published_at DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
        
        rows = conn.execute(text(query), params).fetchall()
        
        intel_list = []
        for r in rows:
            entities = []
            if r[5]:
                try:
                    entities = json.loads(r[5]) if isinstance(r[5], str) else r[5]
                except:
                    pass
            
            intel_list.append({
                "id": r[0], "summary": r[1], "category": r[2], 
                "impact_score": r[3], "relevance_score": r[4],
                "entities": entities, "source_count": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "article_id": r[8], "title": r[9], "url": r[10],
                "published_at": r[11].isoformat() if r[11] else None,
                "source_label": r[12],
                "sources": [{"title": r[9], "url": r[10], "source": r[12]}]
            })
        
        total = conn.execute(text("""
            SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
            WHERE a.published_at >= :cutoff AND i.impact_score >= :min_impact
        """ + (" AND i.category = :category" if category else "")), params).scalar()
        
        return jsonify({"intel": intel_list, "total": total, "limit": limit, "offset": offset})

@app.route("/api/tubi/intel")
@cached(ttl_seconds=60)
def api_tubi_intel():
    limit = request.args.get('limit', 50, type=int)
    
    with engine.connect() as conn:
        cutoff = datetime(2025, 1, 1)
        
        rows = conn.execute(text("""
            SELECT i.id, i.summary, i.category, i.impact_score, i.relevance_score,
                   i.entities_json, i.created_at,
                   a.id, a.title, a.url, a.published_at, a.source_label
            FROM intel i
            JOIN articles a ON i.article_id = a.id
            WHERE a.published_at >= :cutoff
              AND (LOWER(a.title) LIKE '%tubi%' OR LOWER(i.summary) LIKE '%tubi%')
            ORDER BY a.published_at DESC
            LIMIT :limit
        """), {"cutoff": cutoff, "limit": limit}).fetchall()
        
        intel_list = []
        for r in rows:
            entities = []
            if r[5]:
                try:
                    entities = json.loads(r[5]) if isinstance(r[5], str) else r[5]
                except:
                    pass
            
            intel_list.append({
                "id": r[0], "summary": r[1], "category": r[2],
                "impact_score": r[3], "relevance_score": r[4],
                "entities": entities, "created_at": r[6].isoformat() if r[6] else None,
                "article_id": r[7], "title": r[8], "url": r[9],
                "published_at": r[10].isoformat() if r[10] else None,
                "source_label": r[11],
                "sources": [{"title": r[8], "url": r[9], "source": r[11]}]
            })
        
        return jsonify({"intel": intel_list, "total": len(intel_list)})

@app.route("/api/tubi/stats")
@cached(ttl_seconds=120)
def api_tubi_stats():
    with engine.connect() as conn:
        cutoff = datetime(2025, 1, 1)
        
        total = conn.execute(text("""
            SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
            WHERE a.published_at >= :cutoff
              AND (LOWER(a.title) LIKE '%tubi%' OR LOWER(i.summary) LIKE '%tubi%')
        """), {"cutoff": cutoff}).scalar()
        
        high_impact = conn.execute(text("""
            SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
            WHERE a.published_at >= :cutoff AND i.impact_score >= 8
              AND (LOWER(a.title) LIKE '%tubi%' OR LOWER(i.summary) LIKE '%tubi%')
        """), {"cutoff": cutoff}).scalar()
        
        return jsonify({
            "total_intel": total,
            "high_impact": high_impact,
            "sources_tracked": 12
        })

@app.route("/api/competitors")
@cached(ttl_seconds=300)
def api_competitors():
    competitors = [
        {"id": "netflix", "name": "Netflix", "tier": "major"},
        {"id": "amazon_prime", "name": "Amazon Prime Video", "tier": "major"},
        {"id": "disney_plus", "name": "Disney+", "tier": "major"},
        {"id": "max", "name": "Max (HBO)", "tier": "major"},
        {"id": "hulu", "name": "Hulu", "tier": "major"},
        {"id": "peacock", "name": "Peacock", "tier": "major"},
        {"id": "paramount_plus", "name": "Paramount+", "tier": "major"},
        {"id": "apple_tv", "name": "Apple TV+", "tier": "major"},
        {"id": "youtube", "name": "YouTube/YouTube TV", "tier": "major"},
        {"id": "pluto_tv", "name": "Pluto TV", "tier": "fast"},
        {"id": "roku_channel", "name": "The Roku Channel", "tier": "fast"},
        {"id": "freevee", "name": "Amazon Freevee", "tier": "fast"},
        {"id": "xumo", "name": "Xumo", "tier": "fast"},
    ]
    
    with engine.connect() as conn:
        cutoff = datetime(2025, 1, 1)
        for comp in competitors:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM intel i JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff AND a.competitor_id = :cid
            """), {"cutoff": cutoff, "cid": comp["id"]}).scalar()
            comp["intel_count"] = count
    
    return jsonify({"competitors": competitors})

@app.route("/api/competitors/<competitor_id>/intel")
@cached(ttl_seconds=60)
def api_competitor_intel(competitor_id: str):
    limit = request.args.get('limit', 50, type=int)
    
    with engine.connect() as conn:
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
            "id": r[0], "summary": r[1], "category": r[2], "impact_score": r[3],
            "title": r[4], "url": r[5], 
            "published_at": r[6].isoformat() if r[6] else None,
            "source_label": r[7]
        } for r in rows]
        
        return jsonify({"intel": intel_list, "competitor_id": competitor_id})

@app.route("/api/last-updated")
@cached(ttl_seconds=60)
def api_last_updated():
    with engine.connect() as conn:
        last_article = conn.execute(text("SELECT MAX(created_at) FROM articles")).scalar()
        last_intel = conn.execute(text("SELECT MAX(created_at) FROM intel")).scalar()
        return jsonify({
            "last_article": last_article.isoformat() if last_article else None,
            "last_intel": last_intel.isoformat() if last_intel else None
        })

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
    return jsonify({"error": "Internal server error"}), 500

# For local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
