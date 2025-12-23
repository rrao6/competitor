#!/usr/bin/env python3
"""
Daily Radar - Streamlined competitive intelligence collection.

Optimized for automated daily runs:
- Fast RSS collection from all sources
- Parallel web search for fresh intel
- Smart deduplication
- Quality filtering
- Supabase storage
"""
import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_db_engine():
    """Get database engine (Supabase or SQLite)."""
    from sqlalchemy import create_engine
    
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url:
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        return create_engine(db_url, pool_pre_ping=True)
    else:
        # Fall back to SQLite
        from pathlib import Path
        db_path = Path(__file__).parent / "data" / "radar.db"
        return create_engine(f"sqlite:///{db_path}")


def fetch_rss_feeds() -> List[Dict[str, Any]]:
    """Fetch all RSS feeds in parallel."""
    import feedparser
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent / "config" / "radar.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Collect all feeds
    feeds = []
    
    # Competitor feeds
    for comp in config.get('competitors', []):
        for feed in comp.get('feeds', []):
            feeds.append({
                'competitor_id': comp['id'],
                'label': feed['label'],
                'url': feed['url']
            })
    
    # Industry feeds
    for feed in config.get('industry_feeds', []):
        feeds.append({
            'competitor_id': 'industry',
            'label': feed['label'],
            'url': feed['url']
        })
    
    # Tubi feeds
    if 'tubi' in config:
        for feed in config['tubi'].get('feeds', []):
            feeds.append({
                'competitor_id': 'tubi',
                'label': feed['label'],
                'url': feed['url']
            })
    
    logger.info(f"Fetching {len(feeds)} RSS feeds...")
    
    articles = []
    errors = 0
    
    def fetch_feed(feed_info):
        try:
            parsed = feedparser.parse(feed_info['url'])
            results = []
            for entry in parsed.entries[:20]:  # Max 20 per feed
                # Parse date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                else:
                    pub_date = datetime.now()
                
                # Filter to 2025+ only
                if pub_date.year < 2025:
                    continue
                
                results.append({
                    'competitor_id': feed_info['competitor_id'],
                    'source': feed_info['label'],
                    'title': entry.get('title', '')[:500],
                    'url': entry.get('link', ''),
                    'summary': entry.get('summary', entry.get('description', ''))[:2000],
                    'published_at': pub_date
                })
            return results
        except Exception as e:
            return []
    
    # Parallel fetch
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_feed, f): f for f in feeds}
        try:
            for future in as_completed(futures, timeout=90):
                try:
                    results = future.result(timeout=10)
                    articles.extend(results)
                except:
                    errors += 1
        except TimeoutError:
            logger.warning(f"Some feeds timed out, continuing with collected articles")
    
    logger.info(f"Collected {len(articles)} articles from RSS ({errors} feed errors)")
    return articles


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles by URL and content hash."""
    seen_urls = set()
    seen_hashes = set()
    unique = []
    
    for article in articles:
        url = article.get('url', '')
        
        # Skip if URL already seen
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        # Check content hash
        content = f"{article.get('title', '')}{article.get('summary', '')}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        
        article['content_hash'] = content_hash
        unique.append(article)
    
    logger.info(f"Deduplicated: {len(articles)} -> {len(unique)} articles")
    return unique


def store_articles(articles: List[Dict], engine) -> Dict[int, int]:
    """Store articles in database, return mapping of index to article_id."""
    from sqlalchemy import text
    
    # Get or create run
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO runs (started_at, status)
            VALUES (NOW(), 'running')
            RETURNING id
        """))
        run_id = result.fetchone()[0]
        conn.commit()
    
    article_ids = {}
    stored = 0
    
    with engine.connect() as conn:
        for idx, article in enumerate(articles):
            try:
                # Check if URL already exists
                result = conn.execute(text(
                    "SELECT id FROM articles WHERE url = :url"
                ), {'url': article['url']})
                existing = result.fetchone()
                
                if existing:
                    article_ids[idx] = existing[0]
                    continue
                
                # Insert new article
                result = conn.execute(text("""
                    INSERT INTO articles (run_id, competitor_id, source_label, title, url, 
                                         published_at, raw_snippet, hash, created_at,
                                         summary, source)
                    VALUES (:run_id, :competitor_id, :source, :title, :url,
                            :published_at, :summary, :hash, NOW(),
                            :summary, :source)
                    RETURNING id
                """), {
                    'run_id': run_id,
                    'competitor_id': article['competitor_id'],
                    'source': article['source'],
                    'title': article['title'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'summary': article.get('summary', ''),
                    'hash': article.get('content_hash', '')
                })
                article_ids[idx] = result.fetchone()[0]
                stored += 1
                
                if stored % 100 == 0:
                    conn.commit()
                    
            except Exception as e:
                pass  # Skip duplicates
        
        conn.commit()
        
        # Update run status
        conn.execute(text("""
            UPDATE runs SET finished_at = NOW(), status = 'success'
            WHERE id = :run_id
        """), {'run_id': run_id})
        conn.commit()
    
    logger.info(f"Stored {stored} new articles (run_id: {run_id})")
    return article_ids


def classify_articles(articles: List[Dict], article_ids: Dict[int, int], engine):
    """Classify articles using GPT-4o-mini."""
    from openai import OpenAI
    from sqlalchemy import text
    
    client = OpenAI()
    
    # Filter to only new articles (those with article_ids)
    to_classify = [(idx, a) for idx, a in enumerate(articles) if idx in article_ids]
    
    # Check which already have intel
    with engine.connect() as conn:
        existing_intel = set()
        for idx, _ in to_classify:
            article_id = article_ids[idx]
            result = conn.execute(text(
                "SELECT 1 FROM intel WHERE article_id = :id"
            ), {'id': article_id})
            if result.fetchone():
                existing_intel.add(idx)
    
    to_classify = [(idx, a) for idx, a in to_classify if idx not in existing_intel]
    
    if not to_classify:
        logger.info("No new articles to classify")
        return
    
    logger.info(f"Classifying {len(to_classify)} articles...")
    
    # Process in batches
    batch_size = 25
    total_intel = 0
    
    for batch_start in range(0, len(to_classify), batch_size):
        batch = to_classify[batch_start:batch_start + batch_size]
        
        # Format for LLM
        articles_text = ""
        for i, (idx, article) in enumerate(batch, 1):
            title = article.get('title', '')[:100]
            summary = article.get('summary', '')[:200]
            articles_text += f"{i}. {title}\n   {summary}\n\n"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Classify streaming/CTV industry articles. 
For each HIGH-IMPACT article (6+), output EXACTLY:
NUM|CATEGORY|IMPACT|RELEVANCE|ENTITIES|SUMMARY

Rules:
- CATEGORY: strategic/product/content/marketing/ai_ads/pricing
- IMPACT/RELEVANCE: 1-10
- SUMMARY: One sentence with SPECIFIC facts (numbers, deals, dates). No vague phrases.
- Skip low-value articles

Articles:
{articles_text}

Output format only, no extra text:"""
                }],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Parse response
            with engine.connect() as conn:
                for line in response.choices[0].message.content.strip().split('\n'):
                    if '|' not in line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) < 6:
                        continue
                    
                    try:
                        num = int(parts[0].strip())
                        if num < 1 or num > len(batch):
                            continue
                        
                        idx, article = batch[num - 1]
                        article_id = article_ids[idx]
                        
                        category = parts[1].strip().lower()
                        impact = float(parts[2].strip())
                        relevance = float(parts[3].strip())
                        entities = parts[4].strip()
                        summary = '|'.join(parts[5:]).strip()
                        
                        # Skip low quality
                        if impact < 5 or len(summary) < 30:
                            continue
                        
                        # Check for Tubi mention
                        is_tubi = 'tubi' in summary.lower() or 'tubi' in article.get('title', '').lower()
                        
                        conn.execute(text("""
                            INSERT INTO intel (article_id, summary, category, impact_score, 
                                             relevance_score, novelty_score, entities_json,
                                             source_count, is_tubi_related, created_at)
                            VALUES (:article_id, :summary, :category, :impact,
                                    :relevance, 1.0, :entities, 1, :is_tubi, NOW())
                        """), {
                            'article_id': article_id,
                            'summary': summary,
                            'category': category,
                            'impact': impact,
                            'relevance': relevance,
                            'entities': json.dumps([e.strip() for e in entities.split(',')]),
                            'is_tubi': is_tubi
                        })
                        total_intel += 1
                        
                    except Exception as e:
                        pass
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
        
        logger.info(f"  Batch {batch_start//batch_size + 1}: {total_intel} intel items")
    
    logger.info(f"Created {total_intel} intel items")


def run_daily():
    """Run daily competitive intelligence collection."""
    start_time = time.time()
    
    print("=" * 60)
    print("COMPETITOR MONITOR - Daily Collection")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Get database
    engine = get_db_engine()
    logger.info("Database connected")
    
    # 2. Fetch RSS feeds
    articles = fetch_rss_feeds()
    
    # 3. Deduplicate
    articles = deduplicate_articles(articles)
    
    # 4. Store articles
    article_ids = store_articles(articles, engine)
    
    # 5. Classify
    classify_articles(articles, article_ids, engine)
    
    # Summary
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print(f"Articles processed: {len(articles)}")
    print(f"New articles stored: {len(article_ids)}")
    
    # Stats
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM intel"))
        print(f"Total intel items: {result.scalar()}")


if __name__ == "__main__":
    run_daily()

