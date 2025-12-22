"""
Agents module - LLM-backed specialized agents for the Radar system.

Includes:
- Core agents (ingestion, understanding, memory, editor)
- Domain specialists (product, content, marketing, AI/ads)
- Swarm components (classifier swarm, search swarm, specialists)
- Orchestrators (smart pipeline, swarm orchestrator v2)
"""

from radar.agents.ingestion import IngestionAgent
from radar.agents.understanding import UnderstandingAgent
from radar.agents.memory import MemoryAgent
from radar.agents.domain import (
    StrategicAgent,
    ProductAgent,
    ContentAgent,
    MarketingAgent,
    AIPlatformAgent,
    run_all_domain_agents,
)
from radar.agents.editor import EditorAgent

# Swarm components
from radar.agents.classifier_swarm import ClassifierSwarm, run_classifier_swarm
from radar.agents.search_swarm import SearchSwarm, run_search_swarm
from radar.agents.critic import CriticAgent

# Specialists
from radar.agents.specialists import (
    ThreatAnalyst,
    OpportunityFinder,
    TrendTracker,
    CompetitorProfiler,
)

# Orchestrators
from radar.agents.orchestrator_v2 import SwarmOrchestrator, run_swarm

__all__ = [
    # Core agents
    "IngestionAgent",
    "UnderstandingAgent",
    "MemoryAgent",
    "EditorAgent",
    # Domain agents
    "StrategicAgent",
    "ProductAgent",
    "ContentAgent",
    "MarketingAgent",
    "AIPlatformAgent",
    "run_all_domain_agents",
    # Swarm
    "ClassifierSwarm",
    "run_classifier_swarm",
    "SearchSwarm",
    "run_search_swarm",
    "CriticAgent",
    # Specialists
    "ThreatAnalyst",
    "OpportunityFinder",
    "TrendTracker",
    "CompetitorProfiler",
    # Orchestrators
    "SwarmOrchestrator",
    "run_swarm",
]

