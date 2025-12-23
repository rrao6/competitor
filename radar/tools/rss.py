"""
RSS feed fetching tool.

Uses feedparser to retrieve and parse RSS/Atom feeds.
Supports parallel fetching for improved performance.
"""
from __future__ import annotations

import hashlib
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import dataclass

import feedparser
from langchain_core.tools import tool

from radar.config import get_config, FeedConfig


# Default settings
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_WORKERS = 10


@dataclass
class ArticleCandidate:
    """A candidate article from an RSS feed or web search."""
    competitor_id: str
    source_label: str
    title: str
    url: str
    published_at: Optional[datetime]
    raw_snippet: str
    hash: str


def compute_article_hash(competitor_id: str, title: str, url: str) -> str:
    """Compute SHA256 hash for deduplication."""
    content = f"{competitor_id}|{title}|{url}"
    return hashlib.sha256(content.encode()).hexdigest()


def parse_published_date(entry: dict) -> Optional[datetime]:
    """Parse published date from feedparser entry."""
    for field in ["published_parsed", "updated_parsed", "created_parsed"]:
        parsed = entry.get(field)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
    return None


def fetch_feed(
    feed_url: str,
    competitor_id: str,
    source_label: str,
    max_items: int = 20,
    filter_keywords: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[ArticleCandidate]:
    """
    Fetch and parse an RSS/Atom feed.
    
    Args:
        feed_url: URL of the RSS feed
        competitor_id: ID of the competitor (or 'industry')
        source_label: Label for this feed source
        max_items: Maximum number of items to return
        filter_keywords: Optional keywords to filter entries
        timeout: Request timeout in seconds
    
    Returns:
        List of ArticleCandidate objects
    """
    candidates = []
    
    try:
        # Set socket timeout
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        
        try:
            request = urllib.request.Request(
                feed_url,
                headers={'User-Agent': 'TubiRadar/1.0 (Competitive Intelligence)'}
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            feed_content = response.read()
            feed = feedparser.parse(feed_content)
        finally:
            socket.setdefaulttimeout(old_timeout)
        
        if feed.bozo and not feed.entries:
            return []
        
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            
            if not title or not link:
                continue
            
            # Extract snippet
            raw_snippet = ""
            for field in ["description", "summary", "content"]:
                content = entry.get(field)
                if content:
                    if isinstance(content, list):
                        content = content[0].get("value", "") if content else ""
                    raw_snippet = str(content)[:2000]
                    break
            
            # Apply keyword filter
            if filter_keywords:
                text_to_check = f"{title} {raw_snippet}".lower()
                if not any(kw.lower() in text_to_check for kw in filter_keywords):
                    continue
            
            published_at = parse_published_date(entry)
            article_hash = compute_article_hash(competitor_id, title, link)
            
            candidates.append(ArticleCandidate(
                competitor_id=competitor_id,
                source_label=source_label,
                title=title,
                url=link,
                published_at=published_at,
                raw_snippet=raw_snippet,
                hash=article_hash,
            ))
    
    except socket.timeout:
        return []
    except urllib.error.URLError:
        return []
    except Exception:
        return []
    
    return candidates


def _fetch_single_feed(feed_info: dict) -> Tuple[str, List[ArticleCandidate]]:
    """
    Fetch a single feed (used by parallel executor).
    
    Args:
        feed_info: Dict with feed configuration
    
    Returns:
        Tuple of (label, candidates)
    """
    candidates = fetch_feed(
        feed_url=feed_info["url"],
        competitor_id=feed_info["competitor_id"],
        source_label=feed_info["label"],
        max_items=feed_info.get("max_items", 20),
        filter_keywords=feed_info.get("filter_keywords"),
        timeout=feed_info.get("timeout", DEFAULT_TIMEOUT),
    )
    return feed_info["label"], candidates


def get_all_feed_configs() -> List[dict]:
    """
    Get all feed configurations from config.
    
    Returns:
        List of feed config dictionaries
    """
    config = get_config()
    feeds = []
    
    timeout = getattr(config.global_config, 'feed_timeout', DEFAULT_TIMEOUT)
    max_items = config.global_config.max_articles_per_feed
    
    # TUBI FEEDS FIRST (our company - high priority)
    if config.tubi and config.tubi.feeds:
        for feed in config.tubi.feeds:
            feeds.append({
                "url": feed.url,
                "competitor_id": "tubi",
                "label": feed.label,
                "max_items": max_items,
                "filter_keywords": feed.filter_keywords if feed.filter_keywords else None,
                "timeout": timeout,
                "is_industry": False,
                "is_tubi": True,  # Mark as Tubi source
            })
    
    # Competitor feeds
    for competitor in config.competitors:
        for feed in competitor.feeds:
            feeds.append({
                "url": feed.url,
                "competitor_id": competitor.id,
                "label": feed.label,
                "max_items": max_items,
                "filter_keywords": feed.filter_keywords if feed.filter_keywords else None,
                "timeout": timeout,
                "is_industry": False,
            })
    
    # Industry feeds
    for feed in config.industry_feeds:
        feeds.append({
            "url": feed.url,
            "competitor_id": "industry",
            "label": feed.label,
            "max_items": max_items,
            "filter_keywords": feed.filter_keywords if feed.filter_keywords else None,
            "timeout": timeout,
            "is_industry": True,
        })
    
    return feeds


def fetch_all_feeds_parallel(
    max_workers: int = DEFAULT_MAX_WORKERS,
    verbose: bool = True,
) -> List[ArticleCandidate]:
    """
    Fetch articles from all configured RSS feeds in parallel.
    
    Args:
        max_workers: Maximum number of concurrent fetches
        verbose: Print progress messages
    
    Returns:
        List of all article candidates
    """
    config = get_config()
    feeds = get_all_feed_configs()
    
    if not feeds:
        if verbose:
            print("  No feeds configured")
        return []
    
    # Get max workers from config or use default
    max_workers = getattr(config.global_config, 'max_concurrent_feeds', max_workers)
    
    all_candidates = []
    successful = 0
    failed = 0
    
    if verbose:
        print(f"  Fetching {len(feeds)} feeds with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all feed fetches
        future_to_feed = {
            executor.submit(_fetch_single_feed, feed): feed 
            for feed in feeds
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_feed):
            feed = future_to_feed[future]
            try:
                label, candidates = future.result()
                if candidates:
                    all_candidates.extend(candidates)
                    successful += 1
                    if verbose:
                        icon = "📰" if feed["is_industry"] else "📡"
                        print(f"    {icon} {label}: {len(candidates)} articles")
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"    ⚠️  {feed['label']}: error - {e}")
    
    if verbose:
        print(f"  ✅ Completed: {successful} feeds, {len(all_candidates)} articles ({failed} feeds failed)")
    
    return all_candidates


def fetch_all_feeds(verbose: bool = True) -> List[ArticleCandidate]:
    """
    Fetch articles from all configured RSS feeds (parallel by default).
    
    Args:
        verbose: Print progress messages
    
    Returns:
        List of all article candidates from all feeds
    """
    return fetch_all_feeds_parallel(verbose=verbose)


@tool
def fetch_rss(feed_label: str) -> List[dict]:
    """
    Fetch articles from a configured RSS feed by its label.
    
    Args:
        feed_label: The label of the feed as defined in config
    
    Returns:
        List of article candidates as dictionaries
    """
    config = get_config()
    
    # Search for the feed in competitors
    for competitor in config.competitors:
        for feed in competitor.feeds:
            if feed.label == feed_label:
                candidates = fetch_feed(
                    feed_url=feed.url,
                    competitor_id=competitor.id,
                    source_label=feed.label,
                    max_items=config.global_config.max_articles_per_feed,
                    filter_keywords=feed.filter_keywords if feed.filter_keywords else None,
                )
                return [vars(c) for c in candidates]
    
    # Search in industry feeds
    for feed in config.industry_feeds:
        if feed.label == feed_label:
            candidates = fetch_feed(
                feed_url=feed.url,
                competitor_id="industry",
                source_label=feed.label,
                max_items=config.global_config.max_articles_per_feed,
                filter_keywords=feed.filter_keywords if feed.filter_keywords else None,
            )
            return [vars(c) for c in candidates]
    
    return []
