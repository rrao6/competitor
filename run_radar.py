#!/usr/bin/env python3
"""
Tubi Radar - Smart Multi-Agent Competitive Intelligence

Run modes:
  python run_radar.py           # Full swarm mode (recommended)
  python run_radar.py --quick   # Quick RSS-only mode
  python run_radar.py --stream  # Continuous streaming mode
  python run_radar.py --legacy  # Legacy smart pipeline mode
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_environment() -> bool:
    """Check that required environment variables are set."""
    from dotenv import load_dotenv
    
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY is not set")
        return False
    return True


def run_smart(reset_db: bool = False) -> dict:
    """
    Run the SMART multi-agent pipeline.
    
    Uses collaborative agents:
    1. Scout - filters relevant articles
    2. Analyst - deep classification
    3. Strategist - pattern recognition
    4. Synthesizer - executive brief
    """
    from radar.database import init_database, reset_database
    from radar.tools.db_tools import create_run, complete_run
    from radar.agents.ingestion import IngestionAgent
    from radar.agents.orchestrator import run_smart_pipeline
    
    if reset_db:
        print("🗑️  Resetting database...")
        reset_database()
    
    init_database()
    
    print("\n" + "=" * 60)
    print("🚀 TUBI RADAR - Smart Multi-Agent Pipeline")
    print("=" * 60)
    
    run_id = create_run()
    print(f"\n📋 Run ID: {run_id}")
    
    try:
        # Phase 1: Ingest
        print("\n📥 INGESTION")
        print("-" * 40)
        ingestion = IngestionAgent()
        ingestion_result = ingestion.run(run_id=run_id, enable_web_search=True)
        
        articles_stored = ingestion_result.get("articles_stored", 0)
        if articles_stored == 0:
            print("⚠️  No new articles found")
            complete_run(run_id, status="success", notes="No new articles")
            return {"run_id": run_id, "status": "no_articles"}
        
        # Get articles for smart pipeline
        from radar.tools.db_tools import get_unprocessed_articles
        articles = get_unprocessed_articles.invoke({"run_id": run_id, "limit": 200})
        
        # Phase 2: Smart Multi-Agent Processing
        result = run_smart_pipeline(run_id, articles)
        
        # Save report
        if result.get("report"):
            reports_dir = Path(__file__).parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            report_path = reports_dir / f"radar-{datetime.utcnow().strftime('%Y-%m-%d')}-run{run_id}.md"
            report_path.write_text(result["report"])
            
            # Update run with report path
            from radar.database import get_session_factory
            from radar.models import Run
            Session = get_session_factory()
            session = Session()
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.report_path = str(report_path)
                session.commit()
            session.close()
            
            print(f"\n📄 Report: {report_path}")
        
        complete_run(run_id, status="success")
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE")
        print(f"   Intel: {result.get('intel_count', 0)} items")
        print(f"   Themes: {len(result.get('themes', []))}")
        print(f"   Threats: {len(result.get('threats', []))}")
        print("=" * 60 + "\n")
        
        return {"run_id": run_id, "status": "success", **result}
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        complete_run(run_id, status="error", notes=str(e))
        raise


def run_swarm_mode(reset_db: bool = False) -> dict:
    """
    Run the full swarm pipeline.
    
    This is the recommended mode - uses:
    - Parallel RSS + web search collection
    - 4-worker classifier swarm
    - Specialist agents (threat, opportunity, trends)
    - Critic feedback loops
    - Vector memory integration
    """
    from radar.database import init_database, reset_database
    from radar.tools.db_tools import create_run, complete_run
    from radar.agents.orchestrator_v2 import run_swarm
    
    if reset_db:
        print("🗑️  Resetting database...")
        reset_database()
    
    init_database()
    
    run_id = create_run()
    
    try:
        # Run the swarm
        state = run_swarm(run_id=run_id, enable_web_search=True)
        
        # Save report
        if state.report:
            reports_dir = Path(__file__).parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            report_path = reports_dir / f"radar-swarm-{datetime.utcnow().strftime('%Y-%m-%d')}-run{run_id}.md"
            report_path.write_text(state.report)
            
            # Update run with report path
            from radar.database import get_session_factory
            from radar.models import Run
            Session = get_session_factory()
            session = Session()
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.report_path = str(report_path)
                session.commit()
            session.close()
            
            print(f"\n📄 Report: {report_path}")
        
        complete_run(run_id, status="success")
        
        return {
            "run_id": run_id,
            "status": "success",
            "articles": state.total_articles,
            "intel": len(state.classified_intel),
            "threats": len(state.threats),
            "opportunities": len(state.opportunities),
            "trends": len(state.trends),
        }
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        complete_run(run_id, status="error", notes=str(e))
        raise


def run_quick() -> dict:
    """Quick run - RSS only, no web search, no specialists."""
    from radar.database import init_database
    from radar.tools.db_tools import create_run, complete_run
    from radar.tools.rss import fetch_all_feeds_parallel
    from radar.agents.classifier_swarm import run_classifier_swarm
    
    init_database()
    
    print("\n" + "=" * 60)
    print("🚀 TUBI RADAR - Quick Mode")
    print("=" * 60)
    
    run_id = create_run()
    
    try:
        print("\n📥 INGESTION (RSS only)")
        articles = fetch_all_feeds_parallel(verbose=True)
        print(f"  Fetched {len(articles)} articles")
        
        if not articles:
            complete_run(run_id, status="success", notes="No new articles")
            return {"status": "no_articles"}
        
        print("\n📊 CLASSIFICATION")
        articles_data = [
            {
                "id": i,
                "title": a.title,
                "url": a.url,
                "raw_snippet": a.raw_snippet,
                "source": getattr(a, 'source_label', getattr(a, 'source', 'unknown')),
                "competitor_id": a.competitor_id,
            }
            for i, a in enumerate(articles)
        ]
        
        intel = run_classifier_swarm(articles_data)
        print(f"  Classified {len(intel)} intel items")
        
        # Simple report
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        report_path = reports_dir / f"radar-quick-{datetime.utcnow().strftime('%Y-%m-%d')}-run{run_id}.md"
        
        report = f"# Tubi Radar Quick Report\n\n"
        report += f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        report += f"**Articles:** {len(articles)}\n"
        report += f"**Intel Items:** {len(intel)}\n\n"
        report += "## Top Intel\n\n"
        
        for item in intel[:20]:
            report += f"- **[{item.competitor}]** {item.summary} (impact: {item.impact})\n"
        
        report_path.write_text(report)
        print(f"\n📄 Report: {report_path}")
        
        complete_run(run_id, status="success")
        print("\n✅ COMPLETE\n")
        return {"status": "success", "intel": len(intel)}
    
    except Exception as e:
        complete_run(run_id, status="error", notes=str(e))
        raise


def run_stream_mode():
    """Run in continuous streaming mode."""
    from radar.database import init_database
    from radar.stream import start_streaming, StreamConfig, StreamEvent
    
    init_database()
    
    print("\n" + "=" * 60)
    print("🔄 TUBI RADAR - Streaming Mode")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    def on_event(event: StreamEvent):
        if event.event_type == "alert":
            print(f"\n🚨 ALERT: {event.data.get('reason')}")
            intel = event.data.get('intel', {})
            print(f"   [{intel.get('competitor')}] {intel.get('title')}")
        elif event.event_type == "new_intel":
            count = event.data.get('count', 0)
            print(f"📊 New intel: {count} items")
    
    config = StreamConfig(
        poll_interval_seconds=900,  # 15 min
        web_search_interval_seconds=3600,  # 1 hour
        alert_threshold_impact=7.0,
    )
    
    engine = start_streaming(config)
    engine.register_callback(on_event)
    
    try:
        # Keep main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⚠️  Stopping...")
        from radar.stream import stop_streaming
        stop_streaming()


def main():
    parser = argparse.ArgumentParser(
        description="Tubi Radar - Competitive Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_radar.py              # Full swarm mode (recommended)
  python run_radar.py --quick      # Quick RSS-only mode
  python run_radar.py --stream     # Continuous streaming mode
  python run_radar.py --legacy     # Legacy smart pipeline mode
  python run_radar.py --reset-db   # Reset database first
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (RSS only, no specialists)")
    parser.add_argument("--stream", action="store_true", help="Continuous streaming mode")
    parser.add_argument("--legacy", action="store_true", help="Use legacy smart pipeline")
    parser.add_argument("--reset-db", action="store_true", help="Reset database before run")
    args = parser.parse_args()
    
    if not check_environment():
        sys.exit(1)
    
    try:
        if args.stream:
            run_stream_mode()
        elif args.quick:
            run_quick()
        elif args.legacy:
            run_smart(reset_db=args.reset_db)
        else:
            # Default: swarm mode
            run_swarm_mode(reset_db=args.reset_db)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
