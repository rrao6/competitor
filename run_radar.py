#!/usr/bin/env python3
"""
Tubi Radar - Competitive Intelligence System

CLI entrypoint for running the multi-agent radar pipeline.

Usage:
    python run_radar.py                    # Full Phase 1 pipeline
    python run_radar.py --phase1           # Phase 1 only (no memory/domain agents)
    python run_radar.py --full             # Full pipeline with all agents
    python run_radar.py --web-search       # Enable web search (Phase 2+)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_environment():
    """Load environment variables and validate configuration."""
    from dotenv import load_dotenv
    import os
    
    # Load .env file if it exists
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Validate OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ ERROR: OPENAI_API_KEY not set or invalid")
        print("   Please set your OpenAI API key in a .env file or environment variable")
        print("   Example: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    return True


def init_database():
    """Initialize the database schema."""
    from radar.database import init_database
    print("📦 Initializing database...")
    init_database()


def run_phase1(run_id: int):
    """
    Run Phase 1 pipeline: Ingestion -> Understanding -> Report
    
    This is the minimal viable pipeline without memory or domain agents.
    """
    from radar.agents.ingestion import run_ingestion
    from radar.agents.understanding import run_understanding
    from radar.agents.editor import run_editor
    from radar.tools.db_tools import complete_run
    
    print("\n" + "=" * 60)
    print("🚀 TUBI RADAR - Phase 1 Pipeline")
    print("=" * 60)
    
    try:
        # Ingestion
        print("\n📥 INGESTION PHASE")
        print("-" * 40)
        ingestion_result = run_ingestion(run_id, enable_web_search=False)
        
        if ingestion_result.get("articles_stored", 0) == 0:
            print("\n⚠️  No new articles found. Generating empty report.")
        
        # Understanding
        print("\n🧠 UNDERSTANDING PHASE")
        print("-" * 40)
        understanding_result = run_understanding(run_id, index_embeddings=False)
        
        # Report
        print("\n📝 REPORT GENERATION PHASE")
        print("-" * 40)
        editor_result = run_editor(run_id)
        
        # Complete run
        complete_run(run_id, status="success")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ RADAR RUN COMPLETED SUCCESSFULLY")
        print("\n📊 Run Summary:")
        print(f"   Run ID: {run_id}")
        print(f"   Articles: {ingestion_result.get('candidates_found', 0)} found, {ingestion_result.get('articles_stored', 0)} stored")
        print(f"   Intel: {understanding_result.get('intel_created', 0)} items classified")
        print(f"   Report: {editor_result.get('report_path', 'N/A')}")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        complete_run(run_id, status="error", notes=str(e))
        print(f"\n❌ Pipeline failed: {e}")
        raise


def run_full_pipeline(
    enable_web_search: bool = False,
    enable_memory: bool = True,
    enable_domain_agents: bool = True,
):
    """
    Run the full LangGraph-orchestrated pipeline.
    """
    from radar.graph import run_radar_workflow
    
    result = run_radar_workflow(
        enable_web_search=enable_web_search,
        enable_memory=enable_memory,
        enable_domain_agents=enable_domain_agents,
    )
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tubi Radar - Competitive Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_radar.py                 # Run Phase 1 pipeline
  python run_radar.py --full          # Run full multi-agent pipeline
  python run_radar.py --phase1        # Explicit Phase 1 (same as default)
  python run_radar.py --init-db       # Initialize database only
        """,
    )
    
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1 pipeline only (ingestion, understanding, report)",
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline with all agents (requires Phase 2 setup)",
    )
    
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search during ingestion",
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory/deduplication phase",
    )
    
    parser.add_argument(
        "--no-domain",
        action="store_true",
        help="Disable domain agent analysis",
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize/reset database and exit",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    init_database()
    
    if args.init_db:
        print("✅ Database initialized successfully")
        return 0
    
    # Determine which pipeline to run
    if args.full:
        # Full LangGraph pipeline
        run_full_pipeline(
            enable_web_search=args.web_search,
            enable_memory=not args.no_memory,
            enable_domain_agents=not args.no_domain,
        )
    else:
        # Phase 1 simple pipeline
        from radar.tools.db_tools import create_run
        run_id = create_run()
        run_phase1(run_id)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

