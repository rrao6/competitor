"""
Understanding Agent (Classifier) for Tubi Radar.

Responsible for converting raw articles into classified intel items
with summaries, categories, and scores.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from radar.agents.base import BaseAgent
from radar.config import get_config
from radar.schemas import ArticleClassification, ArticleClassificationBatch
from radar.tools.db_tools import (
    get_unprocessed_articles,
    store_intel_from_classifications,
)
from radar.tools.vector import embed_intel_batch


class UnderstandingAgent(BaseAgent):
    """
    The Understanding Agent converts articles into intel.
    
    For each article:
    1. Summarizes the key facts concisely
    2. Assigns a category
    3. Scores relevance and impact intelligently
    4. Extracts key entities
    """
    
    agent_role = "understanding_agent"
    system_prompt = """You are an expert competitive intelligence analyst for Tubi, a leading free ad-supported streaming (FAST) service competing with Netflix, Roku, Pluto TV, Peacock, and others.

Your job is to analyze articles and extract actionable competitive intelligence. Be SMART and SELECTIVE.

## SCORING GUIDELINES (BE STRICT)

**Relevance Score (0-10)** - How relevant is this to streaming/CTV/AVOD?
- 0-2: Not relevant (general tech, unrelated industries)
- 3-4: Tangentially relevant (broad entertainment news, celebrity gossip)
- 5-6: Moderately relevant (general streaming industry news)
- 7-8: Highly relevant (direct competitor moves, AVOD/FAST specific)
- 9-10: Critical (directly affects Tubi's market, major competitive shift)

**Impact Score (0-10)** - How significant is this strategically?
- 0-2: Trivial (minor updates, routine announcements)
- 3-4: Low impact (small features, incremental changes)
- 5-6: Moderate (notable moves worth tracking)
- 7-8: High impact (significant strategic moves, major launches)
- 9-10: Critical (market-changing, requires immediate attention)

## CATEGORIES
- `strategic`: M&A, major partnerships, earnings, subscriber/viewer metrics, executive moves, major corporate announcements
- `product`: Platform features, apps, devices, UX changes, technical capabilities
- `content`: Content deals, original productions, library additions, sports/live rights
- `marketing`: Campaigns, brand positioning, promotional partnerships
- `ai_ads`: Ad tech innovations, AI features, targeting capabilities, ad formats, CTV advertising
- `pricing`: Subscription changes, bundle deals, tier modifications
- `noise`: NOT relevant - celebrity news, gossip, reviews, general entertainment not about platforms

## CRITICAL RULES
1. Mark celebrity/talent news as `noise` unless it's about a major exclusive deal
2. Mark show reviews/ratings as `noise` unless announcing major viewership milestones
3. Be GENEROUS with scoring for actual competitor platform moves
4. Focus on WHAT competitors are DOING, not general industry commentary
5. Extract the core strategic insight, not just summarize the headline

## SUMMARY FORMAT
Write 1-2 concise sentences: What happened + Why it matters competitively. No fluff."""

    def __init__(self, batch_size: int = 15, **kwargs):
        """
        Initialize the Understanding Agent.
        
        Args:
            batch_size: Number of articles to process in each LLM call
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
    
    def _build_articles_prompt(self, articles: list[dict]) -> str:
        """Build the prompt for a batch of articles."""
        lines = ["Analyze these articles. Be selective - only high-scoring items matter.\n"]
        
        for i, article in enumerate(articles, 1):
            lines.append(f"---\n**Article {i}** (ID: {article['id']})")
            lines.append(f"Competitor: {article['competitor_id']}")
            lines.append(f"Source: {article['source_label']}")
            lines.append(f"Title: {article['title']}")
            
            snippet = article.get('raw_snippet', '') or ''
            if snippet:
                # Clean and truncate snippet
                snippet = snippet.replace('\n', ' ').strip()[:1200]
                lines.append(f"Content: {snippet}")
            lines.append("")
        
        lines.append("---\nClassify each article. Remember: be strict with scores, generous only for real strategic moves.")
        return "\n".join(lines)
    
    def _classify_batch(self, articles: list[dict]) -> list[ArticleClassification]:
        """
        Classify a batch of articles using the LLM.
        
        Args:
            articles: List of article dictionaries
        
        Returns:
            List of ArticleClassification objects
        """
        if not articles:
            return []
        
        # Get structured LLM
        structured_llm = self.get_structured_llm(ArticleClassificationBatch)
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_articles_prompt(articles)),
        ]
        
        # Get classification
        try:
            result: ArticleClassificationBatch = structured_llm.invoke(messages)
            return result.classifications
        except Exception as e:
            print(f"[UnderstandingAgent] Error classifying batch: {e}")
            return []
    
    def run(
        self,
        run_id: int,
        index_embeddings: bool = True,
    ) -> dict:
        """
        Execute the understanding/classification process.
        
        Args:
            run_id: The current run ID
            index_embeddings: Whether to index intel in vector store
        
        Returns:
            Dictionary with classification results
        """
        config = get_config()
        
        # Get unprocessed articles
        articles = get_unprocessed_articles.invoke({
            "run_id": run_id,
            "limit": config.global_config.max_items_total,
        })
        
        if not articles:
            print("[UnderstandingAgent] No unprocessed articles found")
            return {
                "articles_processed": 0,
                "intel_created": 0,
                "embeddings_indexed": 0,
            }
        
        print(f"[UnderstandingAgent] Processing {len(articles)} articles...")
        
        all_classifications = []
        
        # Process in batches
        for i in range(0, len(articles), self.batch_size):
            batch = articles[i:i + self.batch_size]
            print(f"[UnderstandingAgent] Classifying batch {i // self.batch_size + 1}...")
            
            classifications = self._classify_batch(batch)
            all_classifications.extend(classifications)
        
        # Store intel
        if all_classifications:
            stored_count = store_intel_from_classifications(all_classifications)
            print(f"[UnderstandingAgent] Stored {stored_count} intel records")
        else:
            stored_count = 0
        
        # Index embeddings (deferred to Memory Agent in full pipeline)
        embeddings_indexed = 0
        
        return {
            "articles_processed": len(articles),
            "intel_created": stored_count,
            "embeddings_indexed": embeddings_indexed,
        }


def run_understanding(run_id: int, index_embeddings: bool = False) -> dict:
    """
    Convenience function to run the Understanding Agent.
    
    Args:
        run_id: The current run ID
        index_embeddings: Whether to index embeddings
    
    Returns:
        Understanding results
    """
    agent = UnderstandingAgent()
    return agent.run(run_id=run_id, index_embeddings=index_embeddings)
