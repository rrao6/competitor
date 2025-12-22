"""
Web search tool using OpenAI's gpt-4o-search-preview model.

Provides real-time web search for competitive intelligence.
"""
from __future__ import annotations

import os
import re
from typing import List
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from radar.config import get_config, get_settings
from radar.tools.rss import ArticleCandidate, compute_article_hash


@dataclass
class SearchResult:
    """A search result from web search."""
    title: str
    url: str
    snippet: str
    source: str


def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Perform a web search using OpenAI's gpt-4o-search-preview model.
    """
    settings = get_settings()
    
    if not settings.openai_api_key:
        return []
    
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={"search_context_size": "medium"},
            messages=[{
                "role": "user",
                "content": f"""Search for the latest news about: {query}

Return exactly {max_results} recent news items in this EXACT format (one per line):

[1] HEADLINE: <exact headline> | SOURCE: <publication name> | URL: <full url> | SUMMARY: <one sentence summary>
[2] HEADLINE: <exact headline> | SOURCE: <publication name> | URL: <full url> | SUMMARY: <one sentence summary>

Focus on news from the last 7 days. Include real URLs from major publications."""
            }],
        )
        
        content = response.choices[0].message.content or ""
        results = []
        
        # Parse structured output
        for line in content.split("\n"):
            line = line.strip()
            if not line or not re.match(r'^\[\d+\]', line):
                continue
            
            # Extract fields
            headline_match = re.search(r'HEADLINE:\s*(.+?)\s*\|', line)
            url_match = re.search(r'URL:\s*(https?://[^\s|]+)', line)
            summary_match = re.search(r'SUMMARY:\s*(.+?)$', line)
            
            if headline_match and url_match:
                results.append(SearchResult(
                    title=headline_match.group(1).strip(),
                    url=url_match.group(1).strip(),
                    snippet=summary_match.group(1).strip() if summary_match else "",
                    source="web_search",
                ))
        
        # If structured parsing failed, try to extract from prose
        if not results:
            results = _parse_prose_response(content, max_results)
        
        return results[:max_results]
        
    except Exception as e:
        print(f"    ⚠️ Web search error: {e}")
        return []


def _parse_prose_response(content: str, max_results: int) -> List[SearchResult]:
    """Extract news items from prose-style response."""
    results = []
    
    # Find URLs in the text
    urls = re.findall(r'https?://[^\s\)]+', content)
    
    # Find headlines (text in bold or before URLs)
    headlines = re.findall(r'\*\*(.+?)\*\*', content)
    
    # Match headlines with URLs
    for i, headline in enumerate(headlines[:max_results]):
        url = urls[i] if i < len(urls) else ""
        if url and headline:
            # Extract surrounding text as snippet
            snippet = ""
            idx = content.find(headline)
            if idx != -1:
                snippet_text = content[idx:idx+300]
                snippet = re.sub(r'\*\*.*?\*\*', '', snippet_text)
                snippet = re.sub(r'\[.*?\]', '', snippet)
                snippet = snippet[:150].strip()
            
            results.append(SearchResult(
                title=headline.strip(),
                url=url.strip(),
                snippet=snippet,
                source="web_search",
            ))
    
    return results


def search_competitor(
    competitor_id: str,
    queries: List[str],
    max_results_per_query: int = 3,
) -> List[ArticleCandidate]:
    """Search for news about a specific competitor."""
    candidates = []
    seen_urls = set()
    
    for query in queries:
        results = search_web(query, max_results=max_results_per_query)
        
        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            
            candidate = ArticleCandidate(
                competitor_id=competitor_id,
                source_label="web_search",
                title=result.title,
                url=result.url,
                published_at=datetime.now(timezone.utc),
                raw_snippet=result.snippet,
                hash=compute_article_hash(competitor_id, result.title, result.url),
            )
            candidates.append(candidate)
    
    return candidates


def search_all_competitors(max_searches: int = 30, verbose: bool = True) -> List[ArticleCandidate]:
    """Search for news about all configured competitors."""
    config = get_config()
    
    if not config.global_config.enable_web_search:
        if verbose:
            print("  ⏭️  Web search disabled")
        return []
    
    all_candidates = []
    seen_urls = set()
    
    # Build search tasks - prioritize competitors without feeds
    search_tasks = []
    
    for comp in config.competitors:
        if not comp.feeds and comp.search_queries:
            for q in comp.search_queries[:2]:
                search_tasks.append((comp.id, q))
    
    for comp in config.competitors:
        if comp.feeds and comp.search_queries:
            search_tasks.append((comp.id, comp.search_queries[0]))
    
    search_tasks = search_tasks[:max_searches]
    
    if verbose:
        print(f"  🔍 Running {len(search_tasks)} web searches...")
    
    # Execute in parallel (limited workers to avoid rate limits)
    def do_search(task):
        comp_id, query = task
        return comp_id, search_web(query, max_results=3)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(do_search, t): t for t in search_tasks}
        
        for future in as_completed(futures):
            try:
                comp_id, results = future.result()
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_candidates.append(ArticleCandidate(
                            competitor_id=comp_id,
                            source_label="web_search",
                            title=r.title,
                            url=r.url,
                            published_at=datetime.now(timezone.utc),
                            raw_snippet=r.snippet,
                            hash=compute_article_hash(comp_id, r.title, r.url),
                        ))
            except Exception:
                pass
    
    if verbose:
        print(f"  ✅ Web search: {len(all_candidates)} results from {len(search_tasks)} queries")
    
    return all_candidates


TRENDING_QUERIES = [
    "streaming wars news today 2025",
    "FAST free streaming channels news",
    "CTV connected TV advertising news",
    "Netflix latest news",
    "Roku platform news updates",
    "Amazon Prime Video Freevee news",
    "Disney+ Hulu streaming news",
    "streaming subscriber growth 2025",
    "AVOD advertising video demand news",
    "streaming originals content deals",
]


def search_trending_topics(max_queries: int = 10, verbose: bool = True) -> List[ArticleCandidate]:
    """Search for trending streaming industry topics."""
    if verbose:
        print(f"  🔥 Searching {max_queries} trending topics...")
    
    all_candidates = []
    seen_urls = set()
    
    for query in TRENDING_QUERIES[:max_queries]:
        results = search_web(query, max_results=3)
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_candidates.append(ArticleCandidate(
                    competitor_id="industry",
                    source_label="web_search_trending",
                    title=r.title,
                    url=r.url,
                    published_at=datetime.now(timezone.utc),
                    raw_snippet=r.snippet,
                    hash=compute_article_hash("industry", r.title, r.url),
                ))
    
    if verbose:
        print(f"  ✅ Trending: {len(all_candidates)} results")
    
    return all_candidates
