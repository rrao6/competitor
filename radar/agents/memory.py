"""
Memory Agent (Archivist) for Tubi Radar.

Responsible for deduplication and computing novelty scores
by comparing new intel against historical data.
"""
from __future__ import annotations

from typing import Optional

from radar.agents.base import BaseAgent
from radar.config import get_config
from radar.tools.db_tools import get_recent_intel_for_dedup, store_novelty_scores
from radar.tools.vector import (
    embed_intel_batch,
    find_duplicates,
    search_similar_intel,
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

    def _compute_novelty(
        self,
        intel_id: int,
        summary: str,
        existing_intel: list[dict],
        similarity_threshold: float = 0.85,
    ) -> dict:
        """
        Compute novelty for a single intel item.
        
        Args:
            intel_id: The intel item's ID
            summary: The intel summary text
            existing_intel: List of existing intel for URL-based dedup
            similarity_threshold: Threshold for semantic duplicate detection
        
        Returns:
            Dict with novelty_score, is_duplicate, duplicate_of
        """
        # Build URL lookup for exact match dedup
        url_to_intel = {}
        for intel in existing_intel:
            if intel.get("url"):
                url_to_intel[intel["url"]] = intel["id"]
        
        # Check for semantic duplicates using vector search
        similar = find_duplicates(
            text=summary,
            threshold=similarity_threshold,
            exclude_ids=[intel_id],
        )
        
        if similar:
            # Found a semantic duplicate
            most_similar = similar[0]
            return {
                "intel_id": intel_id,
                "novelty_score": 0.0,
                "is_duplicate_of": most_similar["intel_id"],
            }
        
        # Compute novelty based on how many similar items exist
        similar_items = search_similar_intel.invoke({
            "text": summary,
            "top_k": 10,
        })
        
        # Filter to recent and relevant
        relevant_similar = [
            s for s in similar_items 
            if s["similarity"] > 0.5 and s["intel_id"] != intel_id
        ]
        
        # Novelty score: fewer similar items = higher novelty
        if not relevant_similar:
            novelty = 1.0  # Completely novel
        else:
            # Average similarity of top matches, inverted
            avg_similarity = sum(s["similarity"] for s in relevant_similar[:5]) / min(len(relevant_similar), 5)
            novelty = max(0.0, 1.0 - avg_similarity)
        
        return {
            "intel_id": intel_id,
            "novelty_score": novelty,
            "is_duplicate_of": None,
        }
    
    def run(
        self,
        run_id: int,
        new_intel_ids: Optional[list[int]] = None,
    ) -> dict:
        """
        Execute the memory/deduplication process.
        
        Args:
            run_id: The current run ID
            new_intel_ids: List of new intel IDs to process (if None, processes all from run)
        
        Returns:
            Dictionary with deduplication results
        """
        config = get_config()
        
        # Get recent intel for comparison
        window_days = config.global_config.dedup.window_days
        existing_intel = get_recent_intel_for_dedup.invoke({
            "window_days": window_days,
        })
        
        print(f"[MemoryAgent] Found {len(existing_intel)} existing intel items for comparison")
        
        if not existing_intel:
            print("[MemoryAgent] No existing intel to compare against")
            return {
                "processed": 0,
                "duplicates_found": 0,
                "indexed": 0,
            }
        
        # Get new intel from this run (those without novelty scores)
        new_intel = [
            i for i in existing_intel
            if i.get("novelty_score") is None
        ]
        
        if not new_intel:
            print("[MemoryAgent] No new intel to process")
            return {
                "processed": 0,
                "duplicates_found": 0,
                "indexed": 0,
            }
        
        print(f"[MemoryAgent] Processing {len(new_intel)} new intel items...")
        
        # Index new intel in vector store
        items_to_embed = []
        for intel in new_intel:
            items_to_embed.append({
                "intel_id": intel["id"],
                "text": intel["summary"],
                "metadata": {
                    "category": intel["category"],
                    "relevance_score": intel.get("relevance_score", 0),
                    "impact_score": intel.get("impact_score", 0),
                },
            })
        
        indexed_count = embed_intel_batch(items_to_embed)
        print(f"[MemoryAgent] Indexed {indexed_count} items in vector store")
        
        # Compute novelty for each new item
        novelty_updates = []
        duplicates_found = 0
        
        for intel in new_intel:
            result = self._compute_novelty(
                intel_id=intel["id"],
                summary=intel["summary"],
                existing_intel=existing_intel,
                similarity_threshold=config.global_config.dedup.similarity_threshold,
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


def run_memory(run_id: int) -> dict:
    """
    Convenience function to run the Memory Agent.
    
    Args:
        run_id: The current run ID
    
    Returns:
        Memory agent results
    """
    agent = MemoryAgent()
    return agent.run(run_id=run_id)

