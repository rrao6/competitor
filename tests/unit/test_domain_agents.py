"""Tests for Domain Agents."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from radar.agents.domain import (
    DomainAgent,
    StrategicAgent,
    ProductAgent,
    ContentAgent,
    MarketingAgent,
    AIPlatformAgent,
    run_all_domain_agents,
)
from radar.schemas import DomainAnnotation, DomainAnnotationBatch


class TestDomainAgentBase:
    """Tests for base DomainAgent class."""
    
    def test_strategic_agent_config(self, mock_config):
        """Test StrategicAgent configuration."""
        agent = StrategicAgent()
        
        assert agent.agent_role == "strategic_agent"
        assert agent.domain_name == "Strategic Updates"
        assert "strategic" in agent.category_filter
    
    def test_product_agent_config(self, mock_config):
        """Test ProductAgent configuration."""
        agent = ProductAgent()
        
        assert agent.agent_role == "product_agent"
        assert "product" in agent.category_filter
    
    def test_content_agent_config(self, mock_config):
        """Test ContentAgent configuration."""
        agent = ContentAgent()
        
        assert agent.agent_role == "content_agent"
        assert "content" in agent.category_filter
    
    def test_marketing_agent_config(self, mock_config):
        """Test MarketingAgent configuration."""
        agent = MarketingAgent()
        
        assert agent.agent_role == "marketing_agent"
        assert "marketing" in agent.category_filter
    
    def test_ai_platform_agent_config(self, mock_config):
        """Test AIPlatformAgent configuration."""
        agent = AIPlatformAgent()
        
        assert agent.agent_role == "ai_platform_agent"
        assert "ai_ads" in agent.category_filter


class TestDomainAgentPromptBuilding:
    """Tests for prompt building."""
    
    def test_build_intel_prompt(self, mock_config):
        """Test building prompt from intel items."""
        agent = ProductAgent()
        
        intel_items = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "category": "product",
                "impact_score": 8.0,
                "relevance_score": 7.0,
                "summary": "Netflix launched new feature",
                "entities": ["Netflix", "Streaming"],
            }
        ]
        
        prompt = agent._build_intel_prompt(intel_items)
        
        assert "Intel #1" in prompt
        assert "netflix" in prompt
        assert "product" in prompt
        assert "Netflix launched new feature" in prompt
        assert "Netflix, Streaming" in prompt
    
    def test_build_intel_prompt_no_entities(self, mock_config):
        """Test prompt building when entities are missing."""
        agent = ProductAgent()
        
        intel_items = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "category": "product",
                "impact_score": 8.0,
                "relevance_score": 7.0,
                "summary": "Netflix news",
                "entities": [],
            }
        ]
        
        prompt = agent._build_intel_prompt(intel_items)
        
        assert "Entities:" not in prompt


class TestDomainAgentAnalysis:
    """Tests for domain agent analysis."""
    
    @patch.object(DomainAgent, "get_structured_llm")
    def test_analyze_batch_success(self, mock_get_llm, mock_config):
        """Test successful batch analysis."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = DomainAnnotationBatch(
            annotations=[
                DomainAnnotation(
                    intel_id=1,
                    so_what="This matters for Tubi",
                    risk_or_opportunity="opportunity",
                    priority="P1",
                    suggested_action="Monitor closely",
                )
            ]
        )
        mock_get_llm.return_value = mock_llm
        
        agent = ProductAgent()
        intel_items = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "category": "product",
                "impact_score": 8.0,
                "relevance_score": 7.0,
                "summary": "Netflix news",
                "entities": [],
            }
        ]
        
        annotations = agent._analyze_batch(intel_items)
        
        assert len(annotations) == 1
        assert annotations[0].intel_id == 1
        assert annotations[0].priority == "P1"
    
    @patch.object(DomainAgent, "get_structured_llm")
    def test_analyze_batch_empty_input(self, mock_get_llm, mock_config):
        """Test analysis with empty input."""
        agent = ProductAgent()
        
        annotations = agent._analyze_batch([])
        
        assert annotations == []
        mock_get_llm.assert_not_called()
    
    @patch.object(DomainAgent, "get_structured_llm")
    def test_analyze_batch_llm_error(self, mock_get_llm, mock_config):
        """Test handling of LLM errors."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm
        
        agent = ProductAgent()
        intel_items = [{"id": 1, "competitor_id": "test", "category": "product", 
                        "impact_score": 5, "relevance_score": 5, "summary": "Test"}]
        
        annotations = agent._analyze_batch(intel_items)
        
        assert annotations == []


class TestDomainAgentRun:
    """Tests for domain agent run method."""
    
    @patch("radar.agents.domain.get_intel_for_domain")
    @patch("radar.agents.domain.store_annotations_from_batch")
    @patch.object(DomainAgent, "_analyze_batch")
    def test_run_with_intel(
        self, mock_analyze, mock_store, mock_get_intel, mock_config
    ):
        """Test run with intel items."""
        mock_get_intel.invoke.return_value = [
            {
                "id": 1,
                "competitor_id": "netflix",
                "category": "product",
                "impact_score": 8.0,
                "relevance_score": 7.0,
                "summary": "Netflix news",
                "entities": [],
            }
        ]
        mock_analyze.return_value = [
            DomainAnnotation(
                intel_id=1,
                so_what="Test",
                risk_or_opportunity="neutral",
                priority="P2",
            )
        ]
        mock_store.return_value = 1
        
        agent = ProductAgent()
        result = agent.run(run_id=1)
        
        assert result["analyzed"] == 1
        assert result["annotations_created"] == 1
    
    @patch("radar.agents.domain.get_intel_for_domain")
    def test_run_no_intel(self, mock_get_intel, mock_config):
        """Test run with no qualifying intel."""
        mock_get_intel.invoke.return_value = []
        
        agent = ProductAgent()
        result = agent.run(run_id=1)
        
        assert result["analyzed"] == 0
        assert result["annotations_created"] == 0


class TestRunAllDomainAgents:
    """Tests for running all domain agents."""
    
    @patch("radar.agents.domain.get_intel_for_domain")
    @patch("radar.agents.domain.store_annotations_from_batch")
    def test_run_all_agents_no_intel(
        self, mock_store, mock_get_intel, mock_config
    ):
        """Test running all domain agents with no intel."""
        mock_get_intel.invoke.return_value = []
        
        results = run_all_domain_agents(run_id=1)
        
        # Should have results from all 5 agents
        assert len(results) == 5
        for agent_name, result in results.items():
            assert result["analyzed"] == 0

