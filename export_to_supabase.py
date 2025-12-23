#!/usr/bin/env python3
"""
Export SQLite data to Supabase PostgreSQL.

Usage:
    1. Set DATABASE_URL environment variable to your Supabase connection string
    2. Run: python export_to_supabase.py

The script will:
    1. Read all data from local SQLite
    2. Connect to Supabase PostgreSQL
    3. Insert all records (skipping duplicates)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_supabase_url() -> str:
    """Get Supabase connection URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("❌ DATABASE_URL environment variable not set")
        print("   Set it to your Supabase connection string:")
        print("   export DATABASE_URL='postgresql://postgres:PASSWORD@HOST:5432/postgres'")
        sys.exit(1)
    
    # Convert postgres:// to postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url


def export_data():
    """Export data from SQLite to Supabase."""
    from radar.database import DEFAULT_DB_PATH
    from radar.models import Run, Article, Intel, Annotation, Report
    
    # Connect to SQLite
    sqlite_url = f"sqlite:///{DEFAULT_DB_PATH}"
    sqlite_engine = create_engine(sqlite_url)
    SqliteSession = sessionmaker(bind=sqlite_engine)
    sqlite_session = SqliteSession()
    
    # Connect to Supabase
    supabase_url = get_supabase_url()
    supabase_engine = create_engine(supabase_url)
    SupabaseSession = sessionmaker(bind=supabase_engine)
    supabase_session = SupabaseSession()
    
    print("🔄 Starting export to Supabase...")
    print()
    
    # Export runs
    print("📋 Exporting runs...")
    runs = sqlite_session.query(Run).all()
    exported_runs = 0
    for run in runs:
        try:
            supabase_session.execute(text("""
                INSERT INTO runs (id, started_at, finished_at, status, notes, report_path)
                VALUES (:id, :started_at, :finished_at, :status, :notes, :report_path)
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": run.id,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "status": run.status,
                "notes": run.notes,
                "report_path": getattr(run, 'report_path', None),
            })
            exported_runs += 1
        except Exception as e:
            print(f"   Run {run.id} skipped: {e}")
    supabase_session.commit()
    print(f"   ✅ Exported {exported_runs} runs")
    
    # Export articles
    print("📰 Exporting articles...")
    articles = sqlite_session.query(Article).all()
    exported_articles = 0
    for article in articles:
        try:
            supabase_session.execute(text("""
                INSERT INTO articles (id, run_id, competitor_id, source_label, title, url, published_at, raw_snippet, hash)
                VALUES (:id, :run_id, :competitor_id, :source_label, :title, :url, :published_at, :raw_snippet, :hash)
                ON CONFLICT (url) DO NOTHING
            """), {
                "id": article.id,
                "run_id": article.run_id,
                "competitor_id": article.competitor_id,
                "source_label": article.source_label,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "raw_snippet": article.raw_snippet,
                "hash": article.hash,
            })
            exported_articles += 1
        except Exception as e:
            pass  # Skip duplicates silently
    supabase_session.commit()
    print(f"   ✅ Exported {exported_articles} articles")
    
    # Export intel
    print("🧠 Exporting intel...")
    intel_items = sqlite_session.query(Intel).all()
    exported_intel = 0
    for intel in intel_items:
        try:
            supabase_session.execute(text("""
                INSERT INTO intel (id, article_id, summary, category, impact_score, relevance_score, novelty_score, source_count, related_urls_json)
                VALUES (:id, :article_id, :summary, :category, :impact_score, :relevance_score, :novelty_score, :source_count, :related_urls_json)
                ON CONFLICT (article_id) DO NOTHING
            """), {
                "id": intel.id,
                "article_id": intel.article_id,
                "summary": intel.summary,
                "category": intel.category,
                "impact_score": intel.impact_score,
                "relevance_score": intel.relevance_score,
                "novelty_score": intel.novelty_score,
                "source_count": intel.source_count,
                "related_urls_json": intel.related_urls_json,
            })
            exported_intel += 1
        except Exception as e:
            pass  # Skip duplicates silently
    supabase_session.commit()
    print(f"   ✅ Exported {exported_intel} intel items")
    
    # Export annotations
    print("📝 Exporting annotations...")
    annotations = sqlite_session.query(Annotation).all()
    exported_annotations = 0
    for ann in annotations:
        try:
            supabase_session.execute(text("""
                INSERT INTO annotations (id, intel_id, agent_role, so_what, risk_opportunity, priority, suggested_action)
                VALUES (:id, :intel_id, :agent_role, :so_what, :risk_opportunity, :priority, :suggested_action)
                ON CONFLICT DO NOTHING
            """), {
                "id": ann.id,
                "intel_id": ann.intel_id,
                "agent_role": ann.agent_role,
                "so_what": ann.so_what,
                "risk_opportunity": ann.risk_opportunity,
                "priority": ann.priority,
                "suggested_action": ann.suggested_action,
            })
            exported_annotations += 1
        except Exception as e:
            pass
    supabase_session.commit()
    print(f"   ✅ Exported {exported_annotations} annotations")
    
    # Close sessions
    sqlite_session.close()
    supabase_session.close()
    
    print()
    print("=" * 50)
    print("✅ EXPORT COMPLETE")
    print("=" * 50)
    print(f"   Runs: {exported_runs}")
    print(f"   Articles: {exported_articles}")
    print(f"   Intel: {exported_intel}")
    print(f"   Annotations: {exported_annotations}")


if __name__ == "__main__":
    export_data()

