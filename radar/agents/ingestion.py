"""
Ingestion Agent (Collector) for Tubi Radar.

Responsible for fetching fresh items from configured RSS feeds
and performing web searches for competitors without RSS feeds.
"""
from __future__ import annotations

from typing import Optional

from radar.agents.base import BaseAgent
from radar.config import get_config
from radar.tools.rss import fetch_all_feeds, ArticleCandidate
from radar.tools.db_tools import store_articles_batch


class IngestionAgent(BaseAgent):
    """
    The Ingestion Agent populates the articles table with fresh items.
    
    Sources:
    1. RSS feeds (parallel fetching)
    2. Web search (for competitors without feeds)
    """
    
    agent_role = "ingestion_agent"
    system_prompt = """You are the Ingestion Agent for Tubi Radar.
Your job is to fetch new items from configured RSS feeds and perform
targeted web searches for competitors without reliable RSS feeds.

Filter out obviously irrelevant items (non-streaming content, unrelated business news).
Use tools to fetch RSS and search the web where configured.
Do not summarize or classify; just ensure we have reasonably clean raw items stored."""

    def run(
        self,
        run_id: int,
        enable_web_search: bool = True,
    ) -> dict:
        """
        Execute the ingestion process.
        
        Args:
            run_id: The current run ID
            enable_web_search: Whether to perform web searches
        
        Returns:
            Dictionary with ingestion results
        """
        config = get_config()
        
        # Fetch all RSS feeds (parallel)
        print("[IngestionAgent] Fetching RSS feeds...")
        rss_candidates = fetch_all_feeds(verbose=True)
        print(f"[IngestionAgent] Found {len(rss_candidates)} articles from RSS")
        
        # Web search for additional coverage
        web_candidates = []
        trending_candidates = []
        
        if enable_web_search and config.global_config.enable_web_search:
            try:
                from radar.tools.web_search import search_all_competitors, search_trending_topics
                
                print("[IngestionAgent] Performing web searches...")
                
                # Search for competitor-specific news
                web_candidates = search_all_competitors(
                    max_searches=config.global_config.max_web_searches,
                    verbose=True,
                )
                
                # Search for trending industry topics
                trending_candidates = search_trending_topics(
                    max_queries=10,
                    verbose=True,
                )
                
            except ImportError:
                print("[IngestionAgent] Web search module not available")
            except Exception as e:
                print(f"[IngestionAgent] Web search error: {e}")
        
        # Combine all candidates
        all_candidates = rss_candidates + web_candidates + trending_candidates
        print(f"[IngestionAgent] Total candidates: {len(all_candidates)} (RSS: {len(rss_candidates)}, Web: {len(web_candidates)}, Trending: {len(trending_candidates)})")
        
        # Store articles (deduplicates by hash)
        stored_count = store_articles_batch(run_id, all_candidates)
        print(f"[IngestionAgent] Stored {stored_count} new articles")
        
        return {
            "candidates_found": len(all_candidates),
            "rss_articles": len(rss_candidates),
            "web_search_articles": len(web_candidates),
            "articles_stored": stored_count,
        }


def run_ingestion(run_id: int, enable_web_search: bool = True) -> dict:
    """
    Convenience function to run the Ingestion Agent.
    
    Args:
        run_id: The current run ID
        enable_web_search: Whether to enable web searches
    
    Returns:
        Ingestion results
    """
    agent = IngestionAgent()
    return agent.run(run_id=run_id, enable_web_search=enable_web_search)
