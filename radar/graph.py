"""
LangGraph workflow orchestration for Tubi Radar.

Defines the multi-agent workflow as a directed graph with:
- Nodes for each agent
- Edges defining the flow
- Shared state for coordination
"""
from __future__ import annotations

from typing import TypedDict, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END


# =============================================================================
# Graph State Definition
# =============================================================================

class RadarState(TypedDict):
    """
    Shared state passed through the workflow.
    
    This is the "blackboard" that all agents can read/write to.
    """
    # Run metadata
    run_id: int
    started_at: str
    
    # Configuration flags
    enable_web_search: bool
    enable_memory: bool
    enable_domain_agents: bool
    
    # Phase results (accumulated through the workflow)
    ingestion_result: Optional[dict]
    understanding_result: Optional[dict]
    memory_result: Optional[dict]
    domain_results: Optional[dict]
    editor_result: Optional[dict]
    
    # Control flags
    has_articles: bool
    has_intel: bool
    error: Optional[str]


# =============================================================================
# Node Functions
# =============================================================================

def start_node(state: RadarState) -> RadarState:
    """Initialize the run."""
    from radar.tools.db_tools import create_run
    
    print("\n" + "=" * 60)
    print("🚀 TUBI RADAR - Starting competitive intelligence run")
    print("=" * 60)
    
    run_id = create_run()
    
    return {
        **state,
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "has_articles": False,
        "has_intel": False,
    }


def ingestion_node(state: RadarState) -> RadarState:
    """Run the Ingestion Agent."""
    from radar.agents.ingestion import IngestionAgent
    
    print("\n📥 INGESTION PHASE")
    print("-" * 40)
    
    try:
        agent = IngestionAgent()
        result = agent.run(
            run_id=state["run_id"],
            enable_web_search=state.get("enable_web_search", False),
        )
        
        has_articles = result.get("articles_stored", 0) > 0
        
        return {
            **state,
            "ingestion_result": result,
            "has_articles": has_articles,
        }
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return {
            **state,
            "error": f"Ingestion failed: {e}",
            "has_articles": False,
        }


def understanding_node(state: RadarState) -> RadarState:
    """Run the Understanding Agent."""
    from radar.agents.understanding import UnderstandingAgent
    
    if not state.get("has_articles"):
        print("\n⏭️  UNDERSTANDING PHASE - Skipped (no articles)")
        return state
    
    print("\n🧠 UNDERSTANDING PHASE")
    print("-" * 40)
    
    try:
        agent = UnderstandingAgent()
        result = agent.run(
            run_id=state["run_id"],
            index_embeddings=False,  # Let Memory Agent handle this
        )
        
        has_intel = result.get("intel_created", 0) > 0
        
        return {
            **state,
            "understanding_result": result,
            "has_intel": has_intel,
        }
    except Exception as e:
        print(f"❌ Understanding failed: {e}")
        return {
            **state,
            "error": f"Understanding failed: {e}",
            "has_intel": False,
        }


def memory_node(state: RadarState) -> RadarState:
    """Run the Memory Agent."""
    from radar.agents.memory import MemoryAgent
    
    if not state.get("enable_memory", True):
        print("\n⏭️  MEMORY PHASE - Disabled")
        return state
    
    if not state.get("has_intel"):
        print("\n⏭️  MEMORY PHASE - Skipped (no intel)")
        return state
    
    print("\n📚 MEMORY PHASE")
    print("-" * 40)
    
    try:
        agent = MemoryAgent()
        result = agent.run(run_id=state["run_id"])
        
        return {
            **state,
            "memory_result": result,
        }
    except Exception as e:
        print(f"⚠️  Memory phase failed (non-fatal): {e}")
        return {
            **state,
            "memory_result": {"error": str(e)},
        }


def domain_agents_node(state: RadarState) -> RadarState:
    """Run all Domain Agents."""
    from radar.agents.domain import run_all_domain_agents
    
    if not state.get("enable_domain_agents", True):
        print("\n⏭️  DOMAIN ANALYSIS - Disabled")
        return state
    
    if not state.get("has_intel"):
        print("\n⏭️  DOMAIN ANALYSIS - Skipped (no intel)")
        return state
    
    print("\n🔍 DOMAIN ANALYSIS PHASE")
    print("-" * 40)
    
    try:
        results = run_all_domain_agents(run_id=state["run_id"])
        
        return {
            **state,
            "domain_results": results,
        }
    except Exception as e:
        print(f"⚠️  Domain analysis failed (non-fatal): {e}")
        return {
            **state,
            "domain_results": {"error": str(e)},
        }


def editor_node(state: RadarState) -> RadarState:
    """Run the Editor Agent."""
    from radar.agents.editor import EditorAgent
    
    print("\n📝 REPORT GENERATION PHASE")
    print("-" * 40)
    
    try:
        agent = EditorAgent()
        result = agent.run(run_id=state["run_id"])
        
        return {
            **state,
            "editor_result": result,
        }
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return {
            **state,
            "error": f"Report generation failed: {e}",
        }


def end_node(state: RadarState) -> RadarState:
    """Finalize the run."""
    from radar.tools.db_tools import complete_run
    
    print("\n" + "=" * 60)
    
    if state.get("error"):
        complete_run(state["run_id"], status="error", notes=state["error"])
        print(f"❌ RADAR RUN COMPLETED WITH ERRORS")
        print(f"   Error: {state['error']}")
    else:
        complete_run(state["run_id"], status="success")
        print("✅ RADAR RUN COMPLETED SUCCESSFULLY")
    
    # Print summary
    print("\n📊 Run Summary:")
    print(f"   Run ID: {state['run_id']}")
    
    if state.get("ingestion_result"):
        ir = state["ingestion_result"]
        print(f"   Articles: {ir.get('candidates_found', 0)} found, {ir.get('articles_stored', 0)} stored")
    
    if state.get("understanding_result"):
        ur = state["understanding_result"]
        print(f"   Intel: {ur.get('intel_created', 0)} items classified")
    
    if state.get("memory_result"):
        mr = state["memory_result"]
        print(f"   Dedup: {mr.get('duplicates_found', 0)} duplicates found")
    
    if state.get("editor_result"):
        er = state["editor_result"]
        print(f"   Report: {er.get('report_path', 'N/A')}")
    
    print("=" * 60 + "\n")
    
    return state


# =============================================================================
# Graph Construction
# =============================================================================

def build_radar_graph() -> StateGraph:
    """
    Build the LangGraph workflow for Tubi Radar.
    
    Graph structure:
    start -> ingestion -> understanding -> memory -> domain_agents -> editor -> end
    
    With conditional skipping based on state.
    """
    # Create the graph
    graph = StateGraph(RadarState)
    
    # Add nodes
    graph.add_node("start", start_node)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("understanding", understanding_node)
    graph.add_node("memory", memory_node)
    graph.add_node("domain_agents", domain_agents_node)
    graph.add_node("editor", editor_node)
    graph.add_node("end", end_node)
    
    # Add edges (linear flow for now)
    graph.add_edge("start", "ingestion")
    graph.add_edge("ingestion", "understanding")
    graph.add_edge("understanding", "memory")
    graph.add_edge("memory", "domain_agents")
    graph.add_edge("domain_agents", "editor")
    graph.add_edge("editor", "end")
    graph.add_edge("end", END)
    
    # Set entry point
    graph.set_entry_point("start")
    
    return graph


def compile_radar_workflow():
    """Compile the radar workflow graph."""
    graph = build_radar_graph()
    return graph.compile()


# =============================================================================
# Workflow Execution
# =============================================================================

def run_radar_workflow(
    enable_web_search: bool = False,
    enable_memory: bool = True,
    enable_domain_agents: bool = True,
) -> RadarState:
    """
    Execute the full Tubi Radar workflow.
    
    Args:
        enable_web_search: Enable web search in ingestion
        enable_memory: Enable memory/dedup phase
        enable_domain_agents: Enable domain agent analysis
    
    Returns:
        Final workflow state
    """
    workflow = compile_radar_workflow()
    
    initial_state: RadarState = {
        "run_id": 0,
        "started_at": "",
        "enable_web_search": enable_web_search,
        "enable_memory": enable_memory,
        "enable_domain_agents": enable_domain_agents,
        "ingestion_result": None,
        "understanding_result": None,
        "memory_result": None,
        "domain_results": None,
        "editor_result": None,
        "has_articles": False,
        "has_intel": False,
        "error": None,
    }
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state

