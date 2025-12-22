"""
Fast Multi-Agent Orchestrator for Tubi Radar.
"""
from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_config, get_settings
from radar.database import get_session_factory
from radar.models import Intel


@dataclass
class IntelItem:
    id: int
    competitor: str
    title: str
    summary: str
    category: str
    impact: float
    relevance: float
    url: str


class FastOrchestrator:
    """Fast, parallel multi-agent processing."""
    
    def __init__(self):
        settings = get_settings()
        config = get_config()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Faster model
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        self.llm_smart = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
    
    def run(self, run_id: int, articles: List[dict]) -> dict:
        """Process articles fast with parallel classification."""
        
        print(f"\n🧠 PROCESSING {len(articles)} articles...")
        
        # Step 1: Parallel batch classification (fast)
        print("  [1/2] Classifying in parallel...")
        intel_items = self._classify_parallel(articles)
        print(f"        → {len(intel_items)} intel items")
        
        # Store to DB
        self._store_intel(run_id, intel_items)
        
        # Step 2: Generate report (one call)
        print("  [2/2] Generating report...")
        report = self._generate_report(intel_items)
        
        return {
            "intel_count": len(intel_items),
            "report": report,
        }
    
    def _classify_parallel(self, articles: List[dict]) -> List[IntelItem]:
        """Classify articles in parallel batches."""
        all_items = []
        batch_size = 20
        
        # Split into batches
        batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._classify_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                try:
                    items = future.result()
                    all_items.extend(items)
                except Exception as e:
                    print(f"        ⚠️ Batch error: {e}")
        
        # Sort by impact
        all_items.sort(key=lambda x: x.impact, reverse=True)
        return all_items
    
    def _classify_batch(self, articles: List[dict]) -> List[IntelItem]:
        """Classify a batch of articles."""
        
        article_text = ""
        for i, a in enumerate(articles):
            title = a.get('title', '')[:100]
            snippet = (a.get('raw_snippet') or '')[:300]
            article_text += f"\n{i+1}. [{a.get('competitor_id')}] {title}\n   {snippet}\n"
        
        prompt = f"""Classify these streaming/CTV industry articles for Tubi competitive intelligence.

For EACH relevant article, output ONE LINE in this format:
NUM|CATEGORY|IMPACT|RELEVANCE|SUMMARY

- NUM: Article number (1, 2, etc.)
- CATEGORY: strategic/product/content/marketing/ai_ads/pricing
- IMPACT: 1-10 (how significant is this move?)
- RELEVANCE: 1-10 (how relevant to Tubi/streaming?)
- SUMMARY: One sentence - what happened and why it matters

SKIP articles that are noise (celebrity gossip, reviews, unrelated tech).
Only include articles with IMPACT >= 5 or RELEVANCE >= 5.

Articles:
{article_text}

Output (one per line, no headers):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_batch(response.content, articles)
        except Exception as e:
            print(f"        ⚠️ Error: {e}")
            return []
    
    def _parse_batch(self, text: str, articles: List[dict]) -> List[IntelItem]:
        """Parse classification output."""
        items = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 5:
                continue
            
            try:
                num = int(parts[0].strip()) - 1
                if num < 0 or num >= len(articles):
                    continue
                
                article = articles[num]
                category = parts[1].strip().lower()
                impact = float(parts[2].strip())
                relevance = float(parts[3].strip())
                summary = "|".join(parts[4:]).strip()
                
                if category not in ["strategic", "product", "content", "marketing", "ai_ads", "pricing"]:
                    category = "strategic"
                
                items.append(IntelItem(
                    id=article.get('id', 0),
                    competitor=article.get('competitor_id', 'unknown'),
                    title=article.get('title', ''),
                    summary=summary,
                    category=category,
                    impact=impact,
                    relevance=relevance,
                    url=article.get('url', ''),
                ))
            except:
                continue
        
        return items
    
    def _store_intel(self, run_id: int, intel_items: List[IntelItem]):
        """Store intel to database."""
        try:
            Session = get_session_factory()
            session = Session()
            
            for item in intel_items:
                existing = session.query(Intel).filter(Intel.article_id == item.id).first()
                if existing:
                    continue
                
                intel = Intel(
                    article_id=item.id,
                    summary=item.summary,
                    category=item.category,
                    relevance_score=item.relevance,
                    impact_score=item.impact,
                    novelty_score=0.8,
                )
                session.add(intel)
            
            session.commit()
            session.close()
        except Exception as e:
            print(f"        ⚠️ DB error: {e}")
    
    def _generate_report(self, intel_items: List[IntelItem]) -> str:
        """Generate executive report."""
        
        if not intel_items:
            return self._empty_report()
        
        # Group by competitor
        by_comp = {}
        for item in intel_items[:40]:
            if item.competitor not in by_comp:
                by_comp[item.competitor] = []
            by_comp[item.competitor].append(item)
        
        intel_text = ""
        for comp, items in sorted(by_comp.items()):
            intel_text += f"\n**{comp.upper()}**:\n"
            for item in items[:5]:
                intel_text += f"- [{item.category}] {item.summary} (Impact: {item.impact})\n"
        
        prompt = f"""Write a sharp, concise competitive intelligence brief for Tubi executives.

Date: {datetime.utcnow().strftime('%B %d, %Y')}

Intel:
{intel_text}

Format:
# Tubi Radar Brief — [Date]

## What Happened This Week
3-5 bullets of the most important moves. Be specific. No fluff.

## By Competitor
Group moves by competitor. One line each.

## Watch List
2-3 things to monitor.

Write like a Bloomberg terminal alert. Executives skim - make every word count."""

        try:
            response = self.llm_smart.invoke([HumanMessage(content=prompt)])
            report = response.content
            report += "\n\n---\n*Generated by Tubi Radar*\n"
            return report
        except Exception as e:
            return f"# Error\n\nFailed: {e}"
    
    def _empty_report(self) -> str:
        return f"""# Tubi Radar Brief — {datetime.utcnow().strftime('%B %d, %Y')}

## What Happened This Week
No significant moves detected.

---
*Generated by Tubi Radar*
"""


def run_smart_pipeline(run_id: int, articles: List[dict]) -> dict:
    """Run the fast pipeline."""
    orchestrator = FastOrchestrator()
    return orchestrator.run(run_id, articles)
