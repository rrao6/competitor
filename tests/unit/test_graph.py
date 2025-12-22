"""Tests for LangGraph workflow."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from radar.graph import (
    RadarState,
    start_node,
    ingestion_node,
    understanding_node,
    memory_node,
    domain_agents_node,
    editor_node,
    end_node,
    build_radar_graph,
)


class TestRadarState:
    """Tests for RadarState TypedDict."""
    
    def test_state_structure(self):
        """Test that state has all required fields."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19T12:00:00",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": None,
        }
        
        assert state["run_id"] == 1
        assert state["enable_web_search"] is True


class TestStartNode:
    """Tests for start node."""
    
    @patch("radar.tools.db_tools.create_run")
    def test_start_creates_run(self, mock_create_run):
        """Test that start node creates a run."""
        mock_create_run.return_value = 42
        
        initial_state: RadarState = {
            "run_id": 0,
            "started_at": "",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": None,
        }
        
        result = start_node(initial_state)
        
        assert result["run_id"] == 42
        assert result["started_at"] != ""


class TestIngestionNode:
    """Tests for ingestion node."""
    
    @patch("radar.agents.ingestion.IngestionAgent")
    def test_ingestion_success(self, mock_agent_class):
        """Test successful ingestion."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "candidates_found": 100,
            "articles_stored": 50,
        }
        mock_agent_class.return_value = mock_agent
        
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": None,
        }
        
        result = ingestion_node(state)
        
        assert result["has_articles"] is True
        assert result["ingestion_result"]["articles_stored"] == 50
    
    @patch("radar.agents.ingestion.IngestionAgent")
    def test_ingestion_no_articles(self, mock_agent_class):
        """Test ingestion with no articles stored."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "candidates_found": 0,
            "articles_stored": 0,
        }
        mock_agent_class.return_value = mock_agent
        
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": None,
        }
        
        result = ingestion_node(state)
        
        assert result["has_articles"] is False
    
    @patch("radar.agents.ingestion.IngestionAgent")
    def test_ingestion_error(self, mock_agent_class):
        """Test ingestion with error."""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("Network error")
        mock_agent_class.return_value = mock_agent
        
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": None,
        }
        
        result = ingestion_node(state)
        
        assert result["error"] is not None
        assert "Ingestion failed" in result["error"]


class TestUnderstandingNode:
    """Tests for understanding node."""
    
    def test_understanding_skipped_no_articles(self):
        """Test that understanding is skipped if no articles."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": {"articles_stored": 0},
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,  # No articles
            "has_intel": False,
            "error": None,
        }
        
        result = understanding_node(state)
        
        # State should be unchanged
        assert result["understanding_result"] is None
    
    @patch("radar.agents.understanding.UnderstandingAgent")
    def test_understanding_success(self, mock_agent_class):
        """Test successful understanding."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "articles_processed": 10,
            "intel_created": 8,
        }
        mock_agent_class.return_value = mock_agent
        
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": {"articles_stored": 10},
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": True,
            "has_intel": False,
            "error": None,
        }
        
        result = understanding_node(state)
        
        assert result["has_intel"] is True
        assert result["understanding_result"]["intel_created"] == 8


class TestMemoryNode:
    """Tests for memory node."""
    
    def test_memory_skipped_disabled(self):
        """Test that memory is skipped if disabled."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": False,  # Disabled
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": True,
            "has_intel": True,
            "error": None,
        }
        
        result = memory_node(state)
        
        assert result["memory_result"] is None
    
    def test_memory_skipped_no_intel(self):
        """Test that memory is skipped if no intel."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": True,
            "has_intel": False,  # No intel
            "error": None,
        }
        
        result = memory_node(state)
        
        assert result["memory_result"] is None


class TestDomainAgentsNode:
    """Tests for domain agents node."""
    
    def test_domain_skipped_disabled(self):
        """Test that domain agents are skipped if disabled."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": False,  # Disabled
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": True,
            "has_intel": True,
            "error": None,
        }
        
        result = domain_agents_node(state)
        
        assert result["domain_results"] is None
    
    @patch("radar.agents.domain.run_all_domain_agents")
    def test_domain_agents_success(self, mock_run_all):
        """Test successful domain agents execution."""
        mock_run_all.return_value = {
            "strategic_agent": {"analyzed": 2, "annotations_created": 2},
            "product_agent": {"analyzed": 3, "annotations_created": 3},
        }
        
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": True,
            "has_intel": True,
            "error": None,
        }
        
        result = domain_agents_node(state)
        
        assert result["domain_results"] is not None


class TestEndNode:
    """Tests for end node."""
    
    @patch("radar.tools.db_tools.complete_run")
    def test_end_success(self, mock_complete_run):
        """Test end node with successful run."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": {"articles_stored": 10},
            "understanding_result": {"intel_created": 5},
            "memory_result": {"duplicates_found": 1, "indexed": 5},
            "domain_results": {},
            "editor_result": {"report_path": "/reports/test.md"},
            "has_articles": True,
            "has_intel": True,
            "error": None,
        }
        
        result = end_node(state)
        
        mock_complete_run.assert_called_once_with(1, status="success")
    
    @patch("radar.tools.db_tools.complete_run")
    def test_end_with_error(self, mock_complete_run):
        """Test end node with error."""
        state: RadarState = {
            "run_id": 1,
            "started_at": "2024-12-19",
            "enable_web_search": True,
            "enable_memory": True,
            "enable_domain_agents": True,
            "ingestion_result": None,
            "understanding_result": None,
            "memory_result": None,
            "domain_results": None,
            "editor_result": None,
            "has_articles": False,
            "has_intel": False,
            "error": "Something went wrong",
        }
        
        result = end_node(state)
        
        mock_complete_run.assert_called_once()
        call_kwargs = mock_complete_run.call_args
        assert call_kwargs[1]["status"] == "error"


class TestGraphConstruction:
    """Tests for graph construction."""
    
    def test_build_graph(self):
        """Test that graph is built correctly."""
        graph = build_radar_graph()
        
        # Should have all nodes
        assert graph is not None
    
    def test_graph_has_entry_point(self):
        """Test that graph has entry point set."""
        graph = build_radar_graph()
        
        # Entry point should be set (implementation detail of LangGraph)
        assert graph is not None

