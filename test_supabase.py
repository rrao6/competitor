#!/usr/bin/env python3
"""
Test script to verify Supabase connection and schema.
Run with: DATABASE_URL='your_url' python test_supabase.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

db_url = os.environ.get('DATABASE_URL', '')
if not db_url:
    print("❌ DATABASE_URL not set!")
    print("Run with: DATABASE_URL='postgresql://...' python test_supabase.py")
    sys.exit(1)

# Fix postgres:// to postgresql://
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)

print(f"🔗 Connecting to: {db_url[:40]}...")

try:
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Test connection
        result = conn.execute(text("SELECT 1")).scalar()
        print(f"✅ Connection successful!")
        
        # Check table counts
        print("\n📊 Table row counts:")
        for table in ['runs', 'articles', 'intel', 'annotations', 'reports']:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"   {table}: {count} rows")
            except Exception as e:
                print(f"   {table}: ERROR - {e}")
        
        # Check intel columns
        print("\n📋 Intel table columns:")
        cols = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'intel' 
            ORDER BY ordinal_position
        """)).fetchall()
        for col in cols:
            print(f"   {col[0]}: {col[1]}")
        
        # Test actual API queries
        print("\n🧪 Testing API queries:")
        
        # Test stats query
        try:
            from datetime import datetime
            cutoff = datetime(2025, 1, 1)
            runs = conn.execute(text("SELECT COUNT(*) FROM runs")).scalar()
            articles = conn.execute(text("SELECT COUNT(*) FROM articles WHERE published_at >= :d"), {"d": cutoff}).scalar()
            print(f"   ✅ Stats query OK - {runs} runs, {articles} articles in 2025")
        except Exception as e:
            print(f"   ❌ Stats query failed: {e}")
        
        # Test intel query (the one that was failing)
        try:
            rows = conn.execute(text("""
                SELECT i.id, i.summary, i.category, i.impact_score, i.relevance_score,
                       i.related_urls_json, i.source_count, i.created_at,
                       a.id as article_id, a.title, a.url, a.published_at, a.source_label,
                       a.competitor_id
                FROM intel i
                JOIN articles a ON i.article_id = a.id
                WHERE a.published_at >= :cutoff AND i.impact_score >= :min_impact
                ORDER BY a.published_at DESC
                LIMIT 5
            """), {"cutoff": cutoff, "min_impact": 0}).fetchall()
            print(f"   ✅ Intel query OK - {len(rows)} results")
            if rows:
                print(f"      Sample: {rows[0][1][:50]}..." if rows[0][1] else "      (no summary)")
        except Exception as e:
            print(f"   ❌ Intel query failed: {e}")
        
        print("\n✅ All tests completed!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

