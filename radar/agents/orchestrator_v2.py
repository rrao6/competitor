"""
Swarm Orchestrator v2 for Tubi Radar.

Connects all swarm components:
- Data Collection (RSS + Search Swarm)
- Parallel Classification
- Specialist Analysis (Threats, Opportunities, Trends)
- Critic Feedback Loop
- Vector Memory Integration
- Final Report Synthesis
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_config, get_settings
from radar.tools.rss import fetch_all_feeds_parallel, ArticleCandidate
from radar.tools.db_tools import store_articles_batch
from radar.database import get_session
from radar.models import Article, Intel
from radar.agents.search_swarm import run_search_swarm
from radar.agents.classifier_swarm import run_classifier_swarm, ClassifiedIntel
from radar.agents.specialists.threat import ThreatAnalyst, ThreatAssessment
from radar.agents.specialists.opportunity import OpportunityFinder, Opportunity
from radar.agents.specialists.trends import TrendTracker, Trend
from radar.agents.specialists.profiler import CompetitorProfiler, CompetitorProfile
from radar.agents.critic import CriticAgent, CritiqueResult
from radar.tools.vector import (
    embed_intel_batch, 
    store_competitor_profile, 
    build_context_for_analysis,
    store_trend,
)


@dataclass
class SwarmState:
    """Shared state for the swarm."""
    run_id: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    
    # Data collection
    rss_articles: List[ArticleCandidate] = field(default_factory=list)
    web_search_articles: List[ArticleCandidate] = field(default_factory=list)
    
    # Classification
    classified_intel: List[ClassifiedIntel] = field(default_factory=list)
    
    # Analysis
    threats: List[ThreatAssessment] = field(default_factory=list)
    opportunities: List[Opportunity] = field(default_factory=list)
    trends: List[Trend] = field(default_factory=list)
    competitor_profiles: Dict[str, CompetitorProfile] = field(default_factory=dict)
    
    # Quality
    critiques: Dict[str, CritiqueResult] = field(default_factory=dict)
    
    # Context
    historical_context: str = ""
    
    # Final output
    report: str = ""
    
    @property
    def total_articles(self) -> int:
        return len(self.rss_articles) + len(self.web_search_articles)
    
    @property
    def all_articles(self) -> List[ArticleCandidate]:
        return self.rss_articles + self.web_search_articles


class SwarmOrchestrator:
    """
    Orchestrator for the multi-agent swarm.
    
    Phases:
    1. Data Collection (parallel RSS + web search)
    2. Classification (4 parallel workers)
    3. Specialist Analysis (parallel threat/opportunity/trend)
    4. Critic Review (feedback loop)
    5. Memory Update (vector store)
    6. Report Synthesis
    """
    
    def __init__(self, run_id: int = 0):
        self.state = SwarmState(run_id=run_id)
        self.config = get_config()
        settings = get_settings()
        
        # Initialize components
        self.threat_analyst = ThreatAnalyst()
        self.opportunity_finder = OpportunityFinder()
        self.trend_tracker = TrendTracker()
        self.profiler = CompetitorProfiler()
        self.critic = CriticAgent()
        
        # Synthesis LLM
        self.synthesis_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    
    def run(self, enable_web_search: bool = True) -> SwarmState:
        """
        Run the full swarm pipeline.
        
        Args:
            enable_web_search: Whether to enable web search
            
        Returns:
            SwarmState with all results
        """
        print(f"\n{'='*60}")
        print(f"  TUBI RADAR SWARM ORCHESTRATOR v2")
        print(f"  Run ID: {self.state.run_id}")
        print(f"{'='*60}\n")
        
        # Phase 1: Data Collection
        self._phase1_data_collection(enable_web_search)
        
        # Phase 2: Classification
        self._phase2_classification()
        
        # Phase 3: Specialist Analysis
        self._phase3_specialist_analysis()
        
        # Phase 4: Critic Review
        self._phase4_critic_review()
        
        # Phase 5: Memory Update
        self._phase5_memory_update()
        
        # Phase 6: Report Synthesis
        self._phase6_synthesis()
        
        # Print summary
        self._print_summary()
        
        return self.state
    
    def _phase1_data_collection(self, enable_web_search: bool):
        """Phase 1: Parallel data collection from RSS and web search."""
        print("PHASE 1: Data Collection")
        print("-" * 40)
        
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # RSS collection
            futures[executor.submit(self._collect_rss)] = "rss"
            
            # Web search collection (if enabled)
            if enable_web_search:
                futures[executor.submit(self._collect_web_search)] = "web_search"
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    count = future.result()
                    print(f"  {source}: {count} articles")
                except Exception as e:
                    print(f"  {source}: ERROR - {e}")
        
        # Store articles to database
        all_articles = self.state.all_articles
        if all_articles:
            stored = store_articles_batch(self.state.run_id, all_articles)
            print(f"  Stored {stored} articles to database")
        
        elapsed = time.time() - start
        print(f"  Total: {self.state.total_articles} articles in {elapsed:.1f}s\n")
    
    def _collect_rss(self) -> int:
        """Collect articles from RSS feeds."""
        articles = fetch_all_feeds_parallel(verbose=False)
        self.state.rss_articles = articles
        return len(articles)
    
    def _collect_web_search(self) -> int:
        """Collect articles from web search swarm."""
        articles = run_search_swarm(max_results_per_query=3)
        self.state.web_search_articles = articles
        return len(articles)
    
    def _phase2_classification(self):
        """Phase 2: Parallel classification using classifier swarm."""
        print("PHASE 2: Classification Swarm")
        print("-" * 40)
        
        start = time.time()
        
        # Convert ArticleCandidate to dict for classifier
        articles_data = []
        for i, a in enumerate(self.state.all_articles):
            articles_data.append({
                "id": i,
                "title": a.title,
                "url": a.url,
                "raw_snippet": a.raw_snippet,
                "source": getattr(a, 'source_label', getattr(a, 'source', 'unknown')),
                "competitor_id": a.competitor_id,
            })
        
        # Run classifier swarm
        intel = run_classifier_swarm(articles_data)
        self.state.classified_intel = intel
        
        # Store intel to database
        stored_intel = self._store_intel_to_db(intel)
        
        elapsed = time.time() - start
        print(f"  Classified {len(intel)} intel items from {len(articles_data)} articles")
        print(f"  Stored {stored_intel} intel items to database")
        print(f"  Time: {elapsed:.1f}s\n")
        
        # Category breakdown
        categories = {}
        for item in intel:
            categories[item.category] = categories.get(item.category, 0) + 1
        print(f"  Categories: {categories}\n")
    
    def _store_intel_to_db(self, intel_items: List[ClassifiedIntel]) -> int:
        """Store classified intel to the database."""
        import json
        stored = 0
        
        with get_session() as session:
            # Get article mapping by URL
            articles = session.query(Article).filter(
                Article.run_id == self.state.run_id
            ).all()
            url_to_article = {a.url: a for a in articles}
            
            for item in intel_items:
                # Find the article
                article = url_to_article.get(item.url)
                if not article:
                    continue
                
                # Check if intel already exists for this article
                existing = session.query(Intel).filter(
                    Intel.article_id == article.id
                ).first()
                if existing:
                    continue
                
                # Get related URLs and source count
                related_urls = getattr(item, 'related_urls', []) or []
                source_count = getattr(item, 'source_count', 1) or 1
                
                intel_record = Intel(
                    article_id=article.id,
                    summary=item.summary,
                    category=item.category,
                    relevance_score=item.relevance,
                    impact_score=item.impact,
                    novelty_score=0.5,  # Default
                    entities_json=json.dumps(item.entities) if item.entities else "[]",
                    source_count=source_count,
                    related_urls_json=json.dumps(related_urls) if related_urls else None,
                )
                session.add(intel_record)
                stored += 1
        
        return stored
    
    def _phase3_specialist_analysis(self):
        """Phase 3: Parallel specialist analysis."""
        print("PHASE 3: Specialist Analysis")
        print("-" * 40)
        
        start = time.time()
        
        # Build historical context from vector memory
        intel_dicts = [{"summary": i.summary, "intel_id": i.article_id} 
                       for i in self.state.classified_intel[:10]]
        self.state.historical_context = build_context_for_analysis(intel_dicts)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # Threat analysis
            futures[executor.submit(
                self.threat_analyst.analyze, 
                self.state.classified_intel
            )] = "threats"
            
            # Opportunity analysis
            futures[executor.submit(
                self.opportunity_finder.analyze, 
                self.state.classified_intel
            )] = "opportunities"
            
            # Trend analysis
            futures[executor.submit(
                self.trend_tracker.analyze, 
                self.state.classified_intel,
                self.state.historical_context
            )] = "trends"
            
            # Competitor profiling
            futures[executor.submit(
                self.profiler.build_all_profiles, 
                self.state.classified_intel
            )] = "profiles"
            
            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result()
                    if analysis_type == "threats":
                        self.state.threats = result
                        print(f"  Threats: {len(result)} identified")
                    elif analysis_type == "opportunities":
                        self.state.opportunities = result
                        print(f"  Opportunities: {len(result)} identified")
                    elif analysis_type == "trends":
                        self.state.trends = result
                        print(f"  Trends: {len(result)} identified")
                    elif analysis_type == "profiles":
                        self.state.competitor_profiles = result
                        print(f"  Competitor Profiles: {len(result)} built")
                except Exception as e:
                    print(f"  {analysis_type}: ERROR - {e}")
        
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s\n")
    
    def _phase4_critic_review(self):
        """Phase 4: Critic reviews analysis quality."""
        print("PHASE 4: Critic Review")
        print("-" * 40)
        
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            if self.state.threats:
                futures[executor.submit(
                    self.critic.critique_threats,
                    self.state.threats,
                    self.state.classified_intel
                )] = "threats"
            
            if self.state.opportunities:
                futures[executor.submit(
                    self.critic.critique_opportunities,
                    self.state.opportunities,
                    self.state.classified_intel
                )] = "opportunities"
            
            if self.state.trends:
                futures[executor.submit(
                    self.critic.critique_trends,
                    self.state.trends,
                    self.state.classified_intel
                )] = "trends"
            
            for future in as_completed(futures):
                critique_type = futures[future]
                try:
                    result = future.result()
                    self.state.critiques[critique_type] = result
                    status = "APPROVED" if result.approved else "NEEDS REVISION"
                    print(f"  {critique_type}: {result.quality_level.value} ({status})")
                except Exception as e:
                    print(f"  {critique_type}: ERROR - {e}")
        
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s\n")
    
    def _phase5_memory_update(self):
        """Phase 5: Update vector memory with new intel and profiles."""
        print("PHASE 5: Memory Update")
        print("-" * 40)
        
        start = time.time()
        
        # Index intel items
        intel_items = [
            {
                "intel_id": i,
                "text": intel.summary,
                "metadata": {
                    "competitor": intel.competitor,
                    "category": intel.category,
                    "impact": intel.impact,
                    "relevance": intel.relevance,
                }
            }
            for i, intel in enumerate(self.state.classified_intel)
        ]
        indexed = embed_intel_batch(intel_items)
        print(f"  Indexed {indexed} intel items to vector store")
        
        # Store competitor profiles
        for comp_id, profile in self.state.competitor_profiles.items():
            profile_text = f"{profile.name}: {profile.strategy_focus}. Strengths: {', '.join(profile.strengths)}. Weaknesses: {', '.join(profile.weaknesses)}."
            store_competitor_profile(comp_id, profile_text, {
                "threat_level": profile.threat_level,
                "business_model": profile.business_model,
            })
        print(f"  Stored {len(self.state.competitor_profiles)} competitor profiles")
        
        # Store trends
        for trend in self.state.trends:
            trend_text = f"{trend.name}: {trend.description}. Prediction: {trend.prediction}"
            store_trend(
                f"{trend.trend_id}_{datetime.now().strftime('%Y%m%d')}",
                trend_text,
                {
                    "category": trend.category,
                    "direction": trend.direction,
                    "strength": trend.strength,
                }
            )
        print(f"  Stored {len(self.state.trends)} trends")
        
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s\n")
    
    def _phase6_synthesis(self):
        """Phase 6: Synthesize final executive brief."""
        print("PHASE 6: Report Synthesis")
        print("-" * 40)
        
        start = time.time()
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt()
        
        try:
            response = self.synthesis_llm.invoke([HumanMessage(content=prompt)])
            self.state.report = response.content
            print(f"  Generated executive brief")
        except Exception as e:
            print(f"  ERROR: {e}")
            self.state.report = self._build_fallback_report()
        
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s\n")
    
    def _build_synthesis_prompt(self) -> str:
        """Build the synthesis prompt for a comprehensive executive brief."""
        # High-impact intel summary
        top_intel = sorted(self.state.classified_intel, key=lambda x: x.impact, reverse=True)[:10]
        intel_text = ""
        for item in top_intel:
            source_count = getattr(item, 'source_count', 1) or 1
            source_indicator = f"[{source_count} sources]" if source_count > 1 else ""
            intel_text += f"\n- [{item.competitor.upper()}] {item.summary} {source_indicator}"
        
        # Top threats with details
        threats_text = ""
        for t in self.state.threats[:6]:
            threats_text += f"\n### {t.competitor.upper()}: {t.threat_type}"
            threats_text += f"\n{t.description}"
            threats_text += f"\n- Severity: {t.severity}/10"
            threats_text += f"\n- Timeframe: {t.timeframe}"
            threats_text += f"\n- Defensive action: {t.defensive_action}\n"
        
        # Top opportunities with details
        opps_text = ""
        for o in self.state.opportunities[:5]:
            opps_text += f"\n### {o.title}"
            opps_text += f"\n{o.description}"
            opps_text += f"\n- Potential Value: {o.potential_value}/10"
            opps_text += f"\n- Feasibility: {o.feasibility}/10"
            opps_text += f"\n- Competitor Gap: {o.competitor_gap}"
            action_items = "; ".join(o.action_items) if o.action_items else "N/A"
            opps_text += f"\n- Action Items: {action_items}\n"
        
        # Trends with predictions
        trends_text = ""
        for t in self.state.trends[:6]:
            trends_text += f"\n### {t.name} ({t.direction.upper()})"
            trends_text += f"\n{t.description}"
            trends_text += f"\n- Strength: {t.strength}/10"
            trends_text += f"\n- Prediction: {t.prediction}\n"
        
        # Competitor profiles
        comp_text = ""
        sorted_profiles = sorted(
            self.state.competitor_profiles.items(), 
            key=lambda x: x[1].threat_level, 
            reverse=True
        )
        for comp_id, profile in sorted_profiles[:8]:
            comp_text += f"\n### {profile.name} (Threat Level: {profile.threat_level}/10)"
            comp_text += f"\n- Strategy: {profile.strategy_focus}"
            comp_text += f"\n- Business Model: {profile.business_model}"
            if profile.strengths:
                comp_text += f"\n- Strengths: {', '.join(profile.strengths[:3])}"
            if profile.weaknesses:
                comp_text += f"\n- Weaknesses: {', '.join(profile.weaknesses[:3])}"
            if profile.recent_moves:
                comp_text += f"\n- Recent Moves: {'; '.join(profile.recent_moves[:3])}"
            if profile.opportunity_areas:
                comp_text += f"\n- Opportunity Areas: {', '.join(profile.opportunity_areas[:2])}\n"
            else:
                comp_text += "\n"

        return f"""You are the Chief Strategy Officer for Tubi, a leading free ad-supported streaming platform.
Your job is to synthesize competitive intelligence into a comprehensive executive brief that will be presented to leadership.

CONTEXT: Tubi is a Fox-owned FAST (Free Ad-Supported Streaming TV) platform competing against:
- Premium SVOD: Netflix, Disney+, Max, Peacock, Paramount+, Apple TV+
- Other FAST/AVOD: Roku Channel, Pluto TV, Freevee (shutting down), Xumo
- Tech Giants: YouTube, Amazon Prime Video, Google TV
- Sports: ESPN+, DAZN, Fubo

DATA ANALYZED THIS RUN:
- {self.state.total_articles} total articles from {len(self.state.competitor_profiles)} competitors
- {len(self.state.classified_intel)} significant intel items identified
- {len(self.state.threats)} competitive threats requiring attention
- {len(self.state.opportunities)} market opportunities discovered
- {len(self.state.trends)} industry trends tracked

═══════════════════════════════════════════════════════════════
HIGH-IMPACT INTEL (Ranked by significance):
{intel_text or "No major intel identified"}

═══════════════════════════════════════════════════════════════
COMPETITIVE THREATS:
{threats_text or "No major threats identified"}

═══════════════════════════════════════════════════════════════
MARKET OPPORTUNITIES:
{opps_text or "No opportunities identified"}

═══════════════════════════════════════════════════════════════
INDUSTRY TRENDS:
{trends_text or "No trends identified"}

═══════════════════════════════════════════════════════════════
COMPETITOR PROFILES:
{comp_text or "No competitor profiles built"}

═══════════════════════════════════════════════════════════════

Write a comprehensive executive brief (800-1000 words) for Tubi leadership. Structure it exactly as follows:

# Tubi Competitive Intelligence Brief
*Generated: {self.state.started_at.strftime('%B %d, %Y')}*

## Executive Summary
(3-4 sentences capturing the most critical insights and the single most important takeaway)

## The Big Picture
(2-3 paragraphs synthesizing the overall competitive landscape this week. What's the narrative? What patterns are emerging?)

## Critical Threats Requiring Action
(For each of the top 3 threats:)
### 1. [Threat Name]
- **What:** Brief description
- **Why it matters to Tubi:** Specific impact
- **Recommended action:** Concrete next step
- **Timeline:** Urgent/This Week/This Month

## Strategic Opportunities
(For each of the top 3 opportunities:)
### 1. [Opportunity Name]
- **The opportunity:** Brief description
- **Why now:** Timing rationale
- **How to capture:** Specific tactics
- **Expected impact:** Quantify if possible

## Market Trends to Monitor
(Bullet points on 4-5 key trends with implications for Tubi)

## Competitor Spotlight
(Focus on the 2-3 most active/threatening competitors this week with specific insights)

## This Week's Priority
(Single paragraph: What is the ONE thing Tubi should focus on based on this intelligence?)

---

Be analytical, not descriptive. Provide insights, not just information summaries.
Use specific numbers, competitor names, and concrete recommendations.
Think like a strategy consultant presenting to C-suite executives.
Format in clean markdown with proper headers."""
    
    def _build_fallback_report(self) -> str:
        """Build a fallback report if synthesis fails."""
        return f"""# Tubi Radar Intelligence Brief

## Summary
- Articles analyzed: {self.state.total_articles}
- Intel items: {len(self.state.classified_intel)}
- Threats: {len(self.state.threats)}
- Opportunities: {len(self.state.opportunities)}
- Trends: {len(self.state.trends)}

## Top Threats
{chr(10).join([f"- {t.competitor}: {t.description}" for t in self.state.threats[:3]]) or "None identified"}

## Top Opportunities
{chr(10).join([f"- {o.title}" for o in self.state.opportunities[:3]]) or "None identified"}

## Key Trends
{chr(10).join([f"- {t.name}: {t.prediction}" for t in self.state.trends[:3]]) or "None identified"}
"""
    
    def _print_summary(self):
        """Print final summary."""
        elapsed = (datetime.now() - self.state.started_at).total_seconds()
        
        print(f"{'='*60}")
        print(f"  SWARM COMPLETE")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Articles: {self.state.total_articles}")
        print(f"  Intel: {len(self.state.classified_intel)}")
        print(f"  Threats: {len(self.state.threats)}")
        print(f"  Opportunities: {len(self.state.opportunities)}")
        print(f"  Trends: {len(self.state.trends)}")
        print(f"{'='*60}\n")


def run_swarm(run_id: int = 0, enable_web_search: bool = True) -> SwarmState:
    """
    Convenience function to run the swarm.
    
    Args:
        run_id: Run identifier
        enable_web_search: Whether to enable web search
        
    Returns:
        SwarmState with all results
    """
    orchestrator = SwarmOrchestrator(run_id=run_id)
    return orchestrator.run(enable_web_search=enable_web_search)

