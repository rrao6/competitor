"""
Ingestion Agent (Collector) for Tubi Radar.

Responsible for fetching fresh items from configured RSS feeds
and optionally performing web searches for each competitor.
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
    
    For Phase 1, this is primarily a non-LLM agent that uses Python
    directly to fetch RSS feeds. Web search integration is optional.
    """
    
    agent_role = "ingestion_agent"
    system_prompt = """You are the Ingestion Agent for Tubi Radar.
Your job is to fetch new items from configured RSS feeds and, optionally, 
perform a few targeted web searches for each competitor.

Filter out obviously irrelevant items (non-streaming Roku, unrelated business news, etc.).
Use tools to fetch RSS, and only call web search where configured.
Do not summarize or classify; just ensure we have reasonably clean raw items stored."""

    def run(
        self,
        run_id: int,
        enable_web_search: bool = False,
    ) -> dict:
        """
        Execute the ingestion process.
        
        Args:
            run_id: The current run ID
            enable_web_search: Whether to perform web searches (Phase 2+)
        
        Returns:
            Dictionary with ingestion results
        """
        config = get_config()
        
        # Fetch all RSS feeds
        print(f"[IngestionAgent] Fetching RSS feeds...")
        candidates = fetch_all_feeds()
        print(f"[IngestionAgent] Found {len(candidates)} article candidates")
        
        # Store articles (deduplicates by hash)
        stored_count = store_articles_batch(run_id, candidates)
        print(f"[IngestionAgent] Stored {stored_count} new articles")
        
        # TODO: Phase 2 - Add web search for search_queries
        web_search_count = 0
        if enable_web_search and config.global_config.enable_web_search:
            # This would use OpenAI's web_search tool
            # For now, just log that it's not implemented
            print("[IngestionAgent] Web search not yet implemented (Phase 2)")
        
        return {
            "candidates_found": len(candidates),
            "articles_stored": stored_count,
            "web_searches_performed": web_search_count,
        }


def run_ingestion(run_id: int, enable_web_search: bool = False) -> dict:
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

