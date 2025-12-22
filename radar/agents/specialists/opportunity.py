"""
Opportunity Finder for Tubi Radar.

Identifies gaps competitors are missing, content/feature opportunities,
and prioritizes by feasibility.
"""
from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_settings
from radar.agents.classifier_swarm import ClassifiedIntel


@dataclass
class Opportunity:
    """An identified opportunity for Tubi."""
    intel_id: int
    opportunity_type: str  # content, feature, market, partnership, technology
    title: str
    description: str
    potential_value: int  # 1-10
    feasibility: int  # 1-10
    priority_score: float  # Calculated from value * feasibility
    action_items: List[str]
    competitor_gap: str  # What gap this exploits


class OpportunityFinder:
    """
    Finds opportunities for Tubi based on competitive intel.
    
    Focus areas:
    - Content gaps (genres/types competitors are missing)
    - Feature opportunities (tech competitors haven't built)
    - Market expansion (underserved demographics/regions)
    - Partnership possibilities
    """
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    
    def analyze(self, intel_items: List[ClassifiedIntel]) -> List[Opportunity]:
        """
        Find opportunities based on intel.
        
        Args:
            intel_items: Classified intel from the classifier swarm
            
        Returns:
            List of opportunities, sorted by priority
        """
        if not intel_items:
            return []
        
        # Focus on items that reveal competitor moves
        relevant = [i for i in intel_items if i.category in ["product", "content", "strategic"]]
        if not relevant:
            relevant = intel_items[:20]
        
        # Build prompt
        intel_text = ""
        for i, item in enumerate(relevant[:25]):
            intel_text += f"\n{i+1}. [{item.competitor}|{item.category}] {item.summary}"
        
        prompt = f"""You are a strategic opportunity analyst for Tubi, the #1 free streaming service (AVOD/FAST).

Tubi's strengths: Free to users, ad-supported, massive content library, strong mobile presence, owned by Fox.

Analyze these competitor moves and identify opportunities for Tubi:

{intel_text}

For each OPPORTUNITY (not threat), output one line:
NUM|TYPE|VALUE|FEASIBILITY|OPPORTUNITY|GAP|ACTION1;ACTION2;ACTION3

- NUM: Intel number (or 0 for industry-wide opportunity)
- TYPE: content/feature/market/partnership/technology
- VALUE: 1-10 potential business value
- FEASIBILITY: 1-10 how easy to execute
- OPPORTUNITY: What Tubi should do
- GAP: What competitor weakness this exploits
- ACTION1;ACTION2;ACTION3: Specific action items (semicolon-separated)

Focus on actionable opportunities. Output (no headers):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, relevant)
        except Exception as e:
            print(f"        OpportunityFinder error: {e}")
            return []
    
    def _parse_response(self, text: str, intel_items: List[ClassifiedIntel]) -> List[Opportunity]:
        """Parse LLM response into Opportunity objects."""
        opportunities = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            try:
                num = int(parts[0].strip().replace(".", ""))
                intel_id = 0
                if 1 <= num <= len(intel_items):
                    intel_id = intel_items[num - 1].article_id
                
                opp_type = parts[1].strip().lower()
                value = int(parts[2].strip())
                feasibility = int(parts[3].strip())
                title = parts[4].strip()
                gap = parts[5].strip()
                actions = [a.strip() for a in parts[6].split(";") if a.strip()]
                
                valid_types = ["content", "feature", "market", "partnership", "technology"]
                if opp_type not in valid_types:
                    opp_type = "feature"
                
                priority_score = (value * feasibility) / 10.0
                
                opportunities.append(Opportunity(
                    intel_id=intel_id,
                    opportunity_type=opp_type,
                    title=title,
                    description=title,
                    potential_value=min(10, max(1, value)),
                    feasibility=min(10, max(1, feasibility)),
                    priority_score=priority_score,
                    action_items=actions[:3],
                    competitor_gap=gap,
                ))
            except Exception:
                continue
        
        # Sort by priority score
        opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return opportunities

