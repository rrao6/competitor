#!/usr/bin/env python3
"""
Migrate data from local SQLite to Supabase PostgreSQL.
"""
import sqlite3
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text

def get_pg_engine():
    """Get PostgreSQL engine for Supabase."""
    db_url = os.environ.get('DATABASE_URL', '')
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    return create_engine(db_url)

def migrate():
    """Migrate all data from SQLite to Supabase."""
    # Connect to local SQLite
    local_conn = sqlite3.connect('data/radar.db')
    local_conn.row_factory = sqlite3.Row
    cursor = local_conn.cursor()
    
    # Connect to Supabase
    pg_engine = get_pg_engine()
    
    print("=" * 60)
    print("MIGRATING TO SUPABASE")
    print("=" * 60)
    
    # Step 1: Clear Supabase tables and drop constraints temporarily
    print("\n1. Clearing Supabase tables...")
    with pg_engine.connect() as conn:
        conn.execute(text('DELETE FROM intel'))
        conn.execute(text('DELETE FROM articles'))
        conn.execute(text('DELETE FROM runs'))
        # Drop unique constraint on URL if it exists
        try:
            conn.execute(text('ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_url_unique'))
            conn.execute(text('ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_url_key'))
        except:
            pass
        conn.commit()
    print("   Done")
    
    # Step 2: Migrate runs
    print("\n2. Migrating runs...")
    cursor.execute('SELECT * FROM runs ORDER BY id')
    runs = cursor.fetchall()
    run_id_map = {}
    
    with pg_engine.connect() as conn:
        for row in runs:
            try:
                result = conn.execute(text('''
                    INSERT INTO runs (started_at, finished_at, status)
                    VALUES (:started_at, :finished_at, :status)
                    RETURNING id
                '''), {
                    'started_at': row['started_at'],
                    'finished_at': row['finished_at'],
                    'status': row['status'] or 'completed'
                })
                run_id_map[row['id']] = result.fetchone()[0]
            except Exception as e:
                print(f"   Run error: {e}")
        conn.commit()
    print(f"   Migrated {len(run_id_map)} runs")
    
    # Step 3: Migrate articles (2025+ only)
    print("\n3. Migrating articles...")
    cursor.execute('''
        SELECT id, run_id, competitor_id, source_label, title, url, 
               published_at, raw_snippet, hash, created_at
        FROM articles 
        WHERE published_at >= '2025-01-01' 
        ORDER BY id
    ''')
    articles = cursor.fetchall()
    article_id_map = {}
    success = 0
    errors = []
    
    with pg_engine.connect() as conn:
        for row in articles:
            old_id = row[0]  # id
            new_run_id = run_id_map.get(row[1], 1)  # run_id
            
            try:
                # Get values by index (SQLite row)
                competitor_id = row[2] or 'unknown'
                source_label = row[3] or 'unknown'
                title = (row[4] or '')[:500]
                url = (row[5] or '')[:2000]
                published_at = row[6]
                raw_snippet = (row[7] or '')[:5000]
                hash_val = row[8] or ''
                created_at = row[9] or datetime.now().isoformat()
                
                result = conn.execute(text('''
                    INSERT INTO articles (run_id, competitor_id, source_label, title, url, 
                                         published_at, raw_snippet, hash, created_at, 
                                         is_tubi_specific, summary, source)
                    VALUES (:run_id, :competitor_id, :source_label, :title, :url,
                            :published_at, :raw_snippet, :hash, :created_at,
                            :is_tubi_specific, :summary, :source)
                    RETURNING id
                '''), {
                    'run_id': new_run_id,
                    'competitor_id': competitor_id,
                    'source_label': source_label,
                    'title': title,
                    'url': url,
                    'published_at': published_at,
                    'raw_snippet': raw_snippet,
                    'hash': hash_val,
                    'created_at': created_at,
                    'is_tubi_specific': 'tubi' in title.lower(),
                    'summary': raw_snippet,
                    'source': source_label,
                })
                article_id_map[old_id] = result.fetchone()[0]
                success += 1
                
                if success % 500 == 0:
                    conn.commit()
                    print(f"   Progress: {success} articles...")
                    
            except Exception as e:
                errors.append(str(e)[:100])
                conn.rollback()
                
        conn.commit()
    print(f"   Migrated {success} articles")
    if errors[:3]:
        print(f"   Sample errors: {errors[:3]}")
    
    # Step 4: Migrate intel
    print("\n4. Migrating intel...")
    
    # Get intel columns first
    cursor.execute('PRAGMA table_info(intel)')
    intel_cols = [row[1] for row in cursor.fetchall()]
    print(f"   Intel columns: {intel_cols}")
    
    cursor.execute('SELECT * FROM intel ORDER BY id')
    intel_items = cursor.fetchall()
    intel_success = 0
    intel_errors = []
    
    with pg_engine.connect() as conn:
        for row in intel_items:
            # Access by index based on columns
            row_dict = dict(zip(intel_cols, row))
            old_article_id = row_dict.get('article_id')
            new_article_id = article_id_map.get(old_article_id)
            
            if not new_article_id:
                continue
            
            try:
                entities = row_dict.get('entities_json') or '[]'
                related_urls = row_dict.get('related_urls_json') or '[]'
                summary = row_dict.get('summary') or ''
                
                conn.execute(text('''
                    INSERT INTO intel (article_id, summary, category, impact_score, relevance_score,
                                       novelty_score, entities_json, source_count, related_urls_json,
                                       is_tubi_related, is_duplicate_of, created_at)
                    VALUES (:article_id, :summary, :category, :impact_score, :relevance_score,
                            :novelty_score, :entities_json, :source_count, :related_urls_json,
                            :is_tubi_related, :is_duplicate_of, :created_at)
                '''), {
                    'article_id': new_article_id,
                    'summary': summary,
                    'category': row_dict.get('category') or 'strategic',
                    'impact_score': float(row_dict.get('impact_score') or 5.0),
                    'relevance_score': float(row_dict.get('relevance_score') or 5.0),
                    'novelty_score': float(row_dict.get('novelty_score') or 1.0),
                    'entities_json': entities,
                    'source_count': int(row_dict.get('source_count') or 1),
                    'related_urls_json': related_urls,
                    'is_tubi_related': 'tubi' in summary.lower(),
                    'is_duplicate_of': row_dict.get('is_duplicate_of'),
                    'created_at': row_dict.get('created_at') or datetime.now().isoformat(),
                })
                intel_success += 1
                
                if intel_success % 200 == 0:
                    conn.commit()
                    print(f"   Progress: {intel_success} intel items...")
                    
            except Exception as e:
                intel_errors.append(str(e)[:100])
                conn.rollback()
                
        conn.commit()
    print(f"   Migrated {intel_success} intel items")
    if intel_errors[:3]:
        print(f"   Sample errors: {intel_errors[:3]}")
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    with pg_engine.connect() as conn:
        result = conn.execute(text('SELECT COUNT(*) FROM runs'))
        print(f"Runs: {result.scalar()}")
        result = conn.execute(text('SELECT COUNT(*) FROM articles'))
        print(f"Articles: {result.scalar()}")
        result = conn.execute(text('SELECT COUNT(*) FROM intel'))
        print(f"Intel: {result.scalar()}")
        result = conn.execute(text("SELECT COUNT(*) FROM intel WHERE is_tubi_related = true"))
        print(f"Tubi Intel: {result.scalar()}")
    
    print("\n✅ Migration complete!")

if __name__ == '__main__':
    migrate()

