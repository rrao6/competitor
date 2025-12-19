"""
RSS feed fetching tool.

Uses feedparser to retrieve and parse RSS/Atom feeds.
"""
from __future__ import annotations

import hashlib
import socket
import urllib.request
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

import feedparser
from langchain_core.tools import tool

from radar.config import get_config, get_competitor_by_id, FeedConfig


# Default timeout for RSS fetches
DEFAULT_TIMEOUT = 10


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
    # Try different date fields
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
    max_items: int = 30,
    filter_keywords: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[ArticleCandidate]:
    """
    Fetch and parse an RSS/Atom feed.
    
    Args:
        feed_url: URL of the RSS feed
        competitor_id: ID of the competitor (or 'industry')
        source_label: Label for this feed source
        max_items: Maximum number of items to return
        filter_keywords: Optional keywords to filter entries (for industry feeds)
        timeout: Request timeout in seconds
    
    Returns:
        List of ArticleCandidate objects
    """
    candidates = []
    
    try:
        # Set socket timeout for feedparser
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        
        try:
            # Create a request with timeout
            request = urllib.request.Request(
                feed_url,
                headers={'User-Agent': 'TubiRadar/1.0 (Competitive Intelligence Bot)'}
            )
            
            # Fetch with timeout
            response = urllib.request.urlopen(request, timeout=timeout)
            feed_content = response.read()
            feed = feedparser.parse(feed_content)
        finally:
            socket.setdefaulttimeout(old_timeout)
        
        if feed.bozo and not feed.entries:
            # Feed parsing failed completely
            print(f"  ⚠️  Feed {source_label}: parsing error")
            return []
        
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            
            if not title or not link:
                continue
            
            # Extract snippet from description or summary
            raw_snippet = ""
            for field in ["description", "summary", "content"]:
                content = entry.get(field)
                if content:
                    if isinstance(content, list):
                        content = content[0].get("value", "") if content else ""
                    raw_snippet = str(content)[:2000]  # Limit snippet size
                    break
            
            # Apply keyword filter for industry feeds
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
        print(f"  ⚠️  Feed {source_label}: timeout after {timeout}s")
        return []
    except urllib.error.URLError as e:
        print(f"  ⚠️  Feed {source_label}: URL error - {e.reason}")
        return []
    except Exception as e:
        print(f"  ⚠️  Feed {source_label}: {type(e).__name__} - {e}")
        return []
    
    return candidates


@tool
def fetch_rss(feed_label: str) -> list[dict]:
    """
    Fetch articles from a configured RSS feed by its label.
    
    Args:
        feed_label: The label of the feed as defined in config (e.g., 'roku_blog', 'deadline')
    
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


def fetch_all_feeds(verbose: bool = True) -> list[ArticleCandidate]:
    """
    Fetch articles from all configured RSS feeds.
    
    Args:
        verbose: Print progress messages
    
    Returns:
        List of all article candidates from all feeds
    """
    config = get_config()
    all_candidates = []
    
    # Get timeout from config or use default
    timeout = getattr(config.global_config, 'feed_timeout', DEFAULT_TIMEOUT)
    
    # Fetch from competitor feeds
    for competitor in config.competitors:
        for feed in competitor.feeds:
            if verbose:
                print(f"  📡 {feed.label}...", end=" ", flush=True)
            candidates = fetch_feed(
                feed_url=feed.url,
                competitor_id=competitor.id,
                source_label=feed.label,
                max_items=config.global_config.max_articles_per_feed,
                filter_keywords=feed.filter_keywords if feed.filter_keywords else None,
                timeout=timeout,
            )
            if verbose:
                print(f"{len(candidates)} articles")
            all_candidates.extend(candidates)
    
    # Fetch from industry feeds
    for feed in config.industry_feeds:
        if verbose:
            print(f"  📰 {feed.label}...", end=" ", flush=True)
        candidates = fetch_feed(
            feed_url=feed.url,
            competitor_id="industry",
            source_label=feed.label,
            max_items=config.global_config.max_articles_per_feed,
            filter_keywords=feed.filter_keywords if feed.filter_keywords else None,
            timeout=timeout,
        )
        if verbose:
            print(f"{len(candidates)} articles")
        all_candidates.extend(candidates)
    
    return all_candidates

