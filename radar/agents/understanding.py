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
    1. Summarizes from Tubi's perspective
    2. Assigns a category
    3. Scores relevance and impact
    4. Extracts key entities
    """
    
    agent_role = "understanding_agent"
    system_prompt = """You are the Understanding Agent for Tubi Radar, a competitive intelligence system for Tubi (a free ad-supported streaming service).

For each article you see, do NOT invent facts. Base your analysis only on the provided content.

For each article, you must:

1. **Summary**: Write 2-3 sentences from Tubi's perspective: what happened? who did it? what does it change for the streaming/AVOD industry?

2. **Category**: Assign exactly ONE category:
   - `product`: Platform features, UX, apps, devices, technology
   - `content`: Shows, movies, content deals, library changes, originals
   - `marketing`: Campaigns, branding, partnerships, promotions
   - `ai_ads`: Advertising technology, AI features, ad products, targeting
   - `pricing`: Subscription tiers, pricing changes, bundle deals
   - `noise`: Irrelevant, off-topic, or not actionable for Tubi

3. **Relevance Score (0-10)**:
   - 0 = Clearly irrelevant to streaming/Tubi
   - 5 = Moderately relevant, general industry news
   - 10 = Extremely relevant to Tubi's product, content, or ad strategy

4. **Impact Score (0-10)**:
   - 0 = Trivia, no strategic significance
   - 5 = Worth noting, moderate industry impact
   - 10 = Must-know strategic move that could affect Tubi's position

5. **Entities**: Extract key entities mentioned (companies, products, platforms, shows, executives).

Be concise and strategic. Focus on what matters for a streaming competitor analyst."""

    def __init__(self, batch_size: int = 10, **kwargs):
        """
        Initialize the Understanding Agent.
        
        Args:
            batch_size: Number of articles to process in each LLM call
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
    
    def _build_articles_prompt(self, articles: list[dict]) -> str:
        """Build the prompt for a batch of articles."""
        lines = ["Analyze the following articles:\n"]
        
        for i, article in enumerate(articles, 1):
            lines.append(f"---\n**Article {i}** (ID: {article['id']})")
            lines.append(f"- Competitor: {article['competitor_id']}")
            lines.append(f"- Source: {article['source_label']}")
            lines.append(f"- Title: {article['title']}")
            lines.append(f"- URL: {article['url']}")
            if article.get('published_at'):
                lines.append(f"- Published: {article['published_at']}")
            lines.append(f"\nContent:\n{article.get('raw_snippet', '[No content available]')[:1500]}")
            lines.append("")
        
        lines.append("---\nProvide your classification for each article.")
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
        
        # Index embeddings
        embeddings_indexed = 0
        if index_embeddings and all_classifications:
            print("[UnderstandingAgent] Indexing embeddings...")
            items_to_embed = []
            for c in all_classifications:
                items_to_embed.append({
                    "intel_id": c.article_id,  # Note: We need the intel ID, not article ID
                    "text": c.summary,
                    "metadata": {
                        "category": c.category,
                        "relevance_score": c.relevance_score,
                        "impact_score": c.impact_score,
                    },
                })
            # TODO: Get actual intel IDs after storage for proper embedding
            # For Phase 1, skip embedding indexing (handled in Phase 2 with Memory Agent)
            print("[UnderstandingAgent] Embedding indexing deferred to Memory Agent (Phase 2)")
        
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

