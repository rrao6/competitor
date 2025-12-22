"""Tests for Understanding Agent."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from radar.agents.understanding import UnderstandingAgent
from radar.schemas import ArticleClassification, ArticleClassificationBatch


class TestUnderstandingAgentPromptBuilding:
    """Tests for Understanding Agent prompt building."""
    
    def test_build_articles_prompt(self, mock_config):
        """Test building prompt from articles."""
        agent = UnderstandingAgent()
        
        articles = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "source_label": "variety",
                "title": "Netflix News",
                "url": "https://example.com/1",
                "published_at": "2024-12-19T12:00:00Z",
                "raw_snippet": "Netflix announced something important today.",
            }
        ]
        
        prompt = agent._build_articles_prompt(articles)
        
        assert "Article 1" in prompt
        assert "ID: 1" in prompt
        assert "netflix" in prompt
        assert "variety" in prompt
        assert "Netflix News" in prompt
    
    def test_build_articles_prompt_truncates_snippet(self, mock_config):
        """Test that long snippets are truncated."""
        agent = UnderstandingAgent()
        
        long_snippet = "A" * 2000  # Very long
        articles = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "source_label": "test",
                "title": "Test",
                "url": "https://example.com",
                "raw_snippet": long_snippet,
            }
        ]
        
        prompt = agent._build_articles_prompt(articles)
        
        # Snippet should be truncated to 1500 chars
        assert len(prompt) < len(long_snippet)
    
    def test_build_articles_prompt_no_published_at(self, mock_config):
        """Test prompt building without published date."""
        agent = UnderstandingAgent()
        
        articles = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "source_label": "test",
                "title": "Test",
                "url": "https://example.com",
                "raw_snippet": "Content",
            }
        ]
        
        prompt = agent._build_articles_prompt(articles)
        
        assert "Published:" not in prompt


class TestUnderstandingAgentClassification:
    """Tests for article classification."""
    
    @patch.object(UnderstandingAgent, "get_structured_llm")
    def test_classify_batch_success(self, mock_get_llm, mock_config):
        """Test successful batch classification."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ArticleClassificationBatch(
            classifications=[
                ArticleClassification(
                    article_id=1,
                    summary="Netflix launched a new feature",
                    category="product",
                    relevance_score=7.0,
                    impact_score=6.0,
                    entities=["Netflix"],
                )
            ]
        )
        mock_get_llm.return_value = mock_llm
        
        agent = UnderstandingAgent()
        articles = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "source_label": "test",
                "title": "Netflix News",
                "url": "https://example.com",
                "raw_snippet": "Netflix announced...",
            }
        ]
        
        classifications = agent._classify_batch(articles)
        
        assert len(classifications) == 1
        assert classifications[0].article_id == 1
        assert classifications[0].category == "product"
    
    @patch.object(UnderstandingAgent, "get_structured_llm")
    def test_classify_batch_empty_input(self, mock_get_llm, mock_config):
        """Test classification with empty input."""
        agent = UnderstandingAgent()
        
        classifications = agent._classify_batch([])
        
        assert classifications == []
        mock_get_llm.assert_not_called()
    
    @patch.object(UnderstandingAgent, "get_structured_llm")
    def test_classify_batch_llm_error(self, mock_get_llm, mock_config):
        """Test handling of LLM errors."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm
        
        agent = UnderstandingAgent()
        articles = [{"id": 1, "competitor_id": "test", "source_label": "test",
                     "title": "Test", "url": "https://example.com", "raw_snippet": "Test"}]
        
        classifications = agent._classify_batch(articles)
        
        assert classifications == []


class TestUnderstandingAgentRun:
    """Tests for Understanding Agent run method."""
    
    @patch("radar.agents.understanding.get_unprocessed_articles")
    @patch("radar.agents.understanding.store_intel_from_classifications")
    @patch.object(UnderstandingAgent, "_classify_batch")
    def test_run_with_articles(
        self, mock_classify, mock_store, mock_get_articles, mock_config
    ):
        """Test run with unprocessed articles."""
        mock_get_articles.invoke.return_value = [
            {"id": 1, "competitor_id": "netflix", "source_label": "test",
             "title": "Test 1", "url": "https://example.com/1", "raw_snippet": "Content 1"},
            {"id": 2, "competitor_id": "disney", "source_label": "test",
             "title": "Test 2", "url": "https://example.com/2", "raw_snippet": "Content 2"},
        ]
        mock_classify.return_value = [
            ArticleClassification(article_id=1, summary="Summary 1", category="product",
                                  relevance_score=7, impact_score=6),
            ArticleClassification(article_id=2, summary="Summary 2", category="content",
                                  relevance_score=8, impact_score=7),
        ]
        mock_store.return_value = 2
        
        agent = UnderstandingAgent()
        result = agent.run(run_id=1, index_embeddings=False)
        
        assert result["articles_processed"] == 2
        assert result["intel_created"] == 2
    
    @patch("radar.agents.understanding.get_unprocessed_articles")
    def test_run_no_articles(self, mock_get_articles, mock_config):
        """Test run with no unprocessed articles."""
        mock_get_articles.invoke.return_value = []
        
        agent = UnderstandingAgent()
        result = agent.run(run_id=1, index_embeddings=False)
        
        assert result["articles_processed"] == 0
        assert result["intel_created"] == 0
    
    @patch("radar.agents.understanding.get_unprocessed_articles")
    @patch("radar.agents.understanding.store_intel_from_classifications")
    @patch.object(UnderstandingAgent, "_classify_batch")
    def test_run_batching(
        self, mock_classify, mock_store, mock_get_articles, mock_config
    ):
        """Test that articles are processed in batches."""
        # Create 25 articles (more than default batch size of 10)
        mock_get_articles.invoke.return_value = [
            {"id": i, "competitor_id": "test", "source_label": "test",
             "title": f"Test {i}", "url": f"https://example.com/{i}", "raw_snippet": "Content"}
            for i in range(25)
        ]
        
        mock_classify.return_value = [
            ArticleClassification(article_id=i, summary=f"Summary {i}", category="product",
                                  relevance_score=5, impact_score=5)
            for i in range(10)  # Each batch returns 10
        ]
        mock_store.return_value = 25
        
        agent = UnderstandingAgent(batch_size=10)
        result = agent.run(run_id=1, index_embeddings=False)
        
        # Should have called classify_batch 3 times (25 / 10 = 3 batches)
        assert mock_classify.call_count == 3


class TestUnderstandingAgentConfig:
    """Tests for Understanding Agent configuration."""
    
    def test_agent_role(self, mock_config):
        """Test agent role is set correctly."""
        agent = UnderstandingAgent()
        
        assert agent.agent_role == "understanding_agent"
    
    def test_custom_batch_size(self, mock_config):
        """Test custom batch size."""
        agent = UnderstandingAgent(batch_size=5)
        
        assert agent.batch_size == 5
    
    def test_model_override(self, mock_config):
        """Test model can be overridden."""
        agent = UnderstandingAgent(model_override="gpt-4")
        
        assert agent.model_name == "gpt-4"

