"""
Memory Agent (Archivist) for Tubi Radar.

Responsible for deduplication and computing novelty scores
by comparing new intel against historical data using ChromaDB.
"""
from __future__ import annotations

from typing import Optional, List

from radar.agents.base import BaseAgent
from radar.config import get_config
from radar.tools.db_tools import (
    get_recent_intel_for_dedup,
    store_novelty_scores,
)


class MemoryAgent(BaseAgent):
    """
    The Memory Agent handles deduplication and novelty scoring.
    
    For each new intel item:
    1. Check if same URL exists in recent intel → duplicate
    2. Check semantic similarity with existing intel → potential duplicate
    3. Compute novelty score based on uniqueness and recency
    """
    
    agent_role = "memory_agent"
    system_prompt = """You are the Memory Agent for Tubi Radar.
Your role is to assess the novelty of new intel items by comparing them to historical intel.
Identify duplicates and compute novelty scores to help prioritize truly new information."""

    def _compute_novelty_simple(
        self,
        intel_id: int,
        summary: str,
        url: str,
        existing_intel: List[dict],
    ) -> dict:
        """
        Compute novelty using simple heuristics (Phase 2A).
        
        Args:
            intel_id: The intel item's ID
            summary: The intel summary text
            url: The article URL
            existing_intel: List of existing intel for comparison
        
        Returns:
            Dict with novelty_score, is_duplicate_of
        """
        # Build URL lookup for exact match dedup
        url_to_intel = {}
        for intel in existing_intel:
            if intel.get("url"):
                url_to_intel[intel["url"]] = intel["id"]
        
        # Check for URL duplicate
        if url in url_to_intel:
            return {
                "intel_id": intel_id,
                "novelty_score": 0.0,
                "is_duplicate_of": url_to_intel[url],
            }
        
        # Simple title/summary similarity check
        summary_lower = summary.lower()
        for existing in existing_intel:
            if existing["id"] == intel_id:
                continue
            
            existing_summary = existing.get("summary", "").lower()
            
            # Check for high overlap (simple word-based)
            summary_words = set(summary_lower.split())
            existing_words = set(existing_summary.split())
            
            if len(summary_words) > 0 and len(existing_words) > 0:
                overlap = len(summary_words & existing_words)
                max_len = max(len(summary_words), len(existing_words))
                similarity = overlap / max_len
                
                if similarity > 0.8:  # Very similar
                    return {
                        "intel_id": intel_id,
                        "novelty_score": 0.1,
                        "is_duplicate_of": existing["id"],
                    }
        
        # Not a duplicate - compute novelty based on topic freshness
        # Higher novelty if fewer similar items exist
        similar_count = 0
        for existing in existing_intel:
            existing_summary = existing.get("summary", "").lower()
            summary_words = set(summary_lower.split())
            existing_words = set(existing_summary.split())
            
            if len(summary_words) > 0 and len(existing_words) > 0:
                overlap = len(summary_words & existing_words)
                max_len = max(len(summary_words), len(existing_words))
                if overlap / max_len > 0.4:
                    similar_count += 1
        
        # Novelty score: fewer similar items = higher novelty
        if similar_count == 0:
            novelty = 1.0
        elif similar_count < 3:
            novelty = 0.8
        elif similar_count < 5:
            novelty = 0.5
        else:
            novelty = 0.3
        
        return {
            "intel_id": intel_id,
            "novelty_score": novelty,
            "is_duplicate_of": None,
        }

    def _compute_novelty_vector(
        self,
        intel_id: int,
        summary: str,
        url: str,
    ) -> dict:
        """
        Compute novelty using ChromaDB vector similarity.
        
        Args:
            intel_id: The intel item's ID
            summary: The intel summary text
            url: The article URL
        
        Returns:
            Dict with novelty_score, is_duplicate_of
        """
        from radar.tools.vector import search_similar_intel, find_duplicates
        
        config = get_config()
        threshold = config.global_config.dedup.similarity_threshold
        
        # Find duplicates using vector similarity
        duplicates = find_duplicates(
            text=summary,
            threshold=threshold,
            exclude_ids=[intel_id],
        )
        
        if duplicates:
            # Found a semantic duplicate
            most_similar = duplicates[0]
            return {
                "intel_id": intel_id,
                "novelty_score": 0.0,
                "is_duplicate_of": most_similar["intel_id"],
            }
        
        # Get similar items for novelty scoring
        similar = search_similar_intel.invoke({
            "text": summary,
            "top_k": 10,
        })
        
        # Filter to relevant similar items
        relevant = [s for s in similar if s["intel_id"] != intel_id and s["similarity"] > 0.5]
        
        # Compute novelty
        if not relevant:
            novelty = 1.0
        else:
            avg_similarity = sum(s["similarity"] for s in relevant[:5]) / min(len(relevant), 5)
            novelty = max(0.1, 1.0 - avg_similarity)
        
        return {
            "intel_id": intel_id,
            "novelty_score": novelty,
            "is_duplicate_of": None,
        }

    def run(
        self,
        run_id: int,
        use_vector_search: bool = True,
    ) -> dict:
        """
        Execute the memory/deduplication process.
        
        Args:
            run_id: The current run ID
            use_vector_search: Use ChromaDB for similarity (vs simple heuristics)
        
        Returns:
            Dictionary with deduplication results
        """
        config = get_config()
        
        # Get recent intel for comparison
        window_days = config.global_config.dedup.window_days
        existing_intel = get_recent_intel_for_dedup.invoke({
            "window_days": window_days,
        })
        
        print(f"[MemoryAgent] Found {len(existing_intel)} existing intel items")
        
        # Get new intel from this run (without novelty scores)
        new_intel = [i for i in existing_intel if i.get("novelty_score") is None]
        
        if not new_intel:
            print("[MemoryAgent] No new intel to process")
            return {
                "processed": 0,
                "duplicates_found": 0,
                "indexed": 0,
            }
        
        print(f"[MemoryAgent] Processing {len(new_intel)} new intel items...")
        
        # Index new intel in vector store if using vector search
        indexed_count = 0
        if use_vector_search:
            try:
                from radar.tools.vector import embed_intel_batch
                
                items_to_embed = [
                    {
                        "intel_id": intel["id"],
                        "text": intel["summary"],
                        "metadata": {
                            "category": intel["category"],
                            "relevance_score": intel.get("relevance_score", 0),
                            "impact_score": intel.get("impact_score", 0),
                        },
                    }
                    for intel in new_intel
                ]
                
                indexed_count = embed_intel_batch(items_to_embed)
                print(f"[MemoryAgent] Indexed {indexed_count} items in vector store")
            except Exception as e:
                print(f"[MemoryAgent] Vector indexing error: {e}")
                use_vector_search = False
        
        # Compute novelty for each new item
        novelty_updates = []
        duplicates_found = 0
        
        for intel in new_intel:
            if use_vector_search:
                try:
                    result = self._compute_novelty_vector(
                        intel_id=intel["id"],
                        summary=intel["summary"],
                        url=intel.get("url", ""),
                    )
                except Exception:
                    result = self._compute_novelty_simple(
                        intel_id=intel["id"],
                        summary=intel["summary"],
                        url=intel.get("url", ""),
                        existing_intel=existing_intel,
                    )
            else:
                result = self._compute_novelty_simple(
                    intel_id=intel["id"],
                    summary=intel["summary"],
                    url=intel.get("url", ""),
                    existing_intel=existing_intel,
                )
            
            novelty_updates.append(result)
            if result.get("is_duplicate_of"):
                duplicates_found += 1
        
        # Store novelty scores
        if novelty_updates:
            store_novelty_scores.invoke({"updates": novelty_updates})
        
        print(f"[MemoryAgent] Found {duplicates_found} duplicates")
        
        return {
            "processed": len(new_intel),
            "duplicates_found": duplicates_found,
            "indexed": indexed_count,
        }


def run_memory(run_id: int, use_vector_search: bool = True) -> dict:
    """
    Convenience function to run the Memory Agent.
    
    Args:
        run_id: The current run ID
        use_vector_search: Use vector similarity for dedup
    
    Returns:
        Memory agent results
    """
    agent = MemoryAgent()
    return agent.run(run_id=run_id, use_vector_search=use_vector_search)


