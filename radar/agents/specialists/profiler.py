"""
Competitor Profiler for Tubi Radar.

Maintains live profiles per competitor, tracks strategy changes,
and stores in vector DB for context.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_settings
from radar.agents.classifier_swarm import ClassifiedIntel


@dataclass
class CompetitorProfile:
    """A competitor profile."""
    competitor_id: str
    name: str
    last_updated: datetime
    business_model: str  # SVOD, AVOD, hybrid, FAST
    strengths: List[str]
    weaknesses: List[str]
    recent_moves: List[str]
    strategy_focus: str
    threat_level: int  # 1-10, relative to Tubi
    opportunity_areas: List[str]
    key_metrics: Dict[str, Any] = field(default_factory=dict)


class CompetitorProfiler:
    """
    Builds and maintains competitor profiles.
    
    Creates rich profiles based on accumulated intel,
    identifying patterns in competitor behavior.
    """
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        
        # Known competitor metadata
        self.competitor_metadata = {
            "netflix": {"name": "Netflix", "model": "SVOD with ads"},
            "disney": {"name": "Disney+/Hulu/ESPN+", "model": "hybrid"},
            "amazon": {"name": "Prime Video/Freevee", "model": "hybrid"},
            "roku": {"name": "Roku/The Roku Channel", "model": "AVOD/FAST"},
            "paramount": {"name": "Paramount+/Pluto TV", "model": "hybrid"},
            "peacock": {"name": "Peacock", "model": "hybrid"},
            "max": {"name": "Max/HBO", "model": "SVOD with ads"},
            "youtube": {"name": "YouTube/YouTube TV", "model": "hybrid"},
            "apple": {"name": "Apple TV+", "model": "SVOD"},
        }
    
    def build_profile(
        self, 
        competitor_id: str,
        intel_items: List[ClassifiedIntel]
    ) -> Optional[CompetitorProfile]:
        """
        Build a profile for a specific competitor.
        
        Args:
            competitor_id: The competitor ID
            intel_items: All classified intel
            
        Returns:
            CompetitorProfile if enough data, else None
        """
        # Filter intel for this competitor
        competitor_intel = [i for i in intel_items if i.competitor.lower() == competitor_id.lower()]
        
        if len(competitor_intel) < 2:
            return None
        
        metadata = self.competitor_metadata.get(competitor_id.lower(), {
            "name": competitor_id,
            "model": "streaming"
        })
        
        # Build prompt
        intel_text = ""
        for i, item in enumerate(competitor_intel[:15]):
            intel_text += f"\n{i+1}. [{item.category}] {item.summary}"
        
        prompt = f"""Analyze {metadata['name']} ({metadata['model']}) based on recent intel.

Recent intel about this competitor:
{intel_text}

Output a structured profile:
STRATEGY: One sentence describing their current strategic focus
THREAT: 1-10 threat level to Tubi (AVOD leader)
STRENGTHS: strength1; strength2; strength3
WEAKNESSES: weakness1; weakness2; weakness3
RECENT_MOVES: move1; move2; move3
OPPORTUNITIES: Area where Tubi can outcompete them

Be specific to the intel provided. Output exactly in this format:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_profile(response.content, competitor_id, metadata)
        except Exception as e:
            print(f"        CompetitorProfiler error for {competitor_id}: {e}")
            return None
    
    def _parse_profile(
        self, 
        text: str, 
        competitor_id: str,
        metadata: Dict[str, str]
    ) -> CompetitorProfile:
        """Parse LLM response into CompetitorProfile."""
        lines = text.strip().split("\n")
        
        strategy = ""
        threat = 5
        strengths = []
        weaknesses = []
        moves = []
        opportunities = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("STRATEGY:"):
                strategy = line.replace("STRATEGY:", "").strip()
            elif line.startswith("THREAT:"):
                try:
                    threat = int(line.replace("THREAT:", "").strip().split()[0])
                except:
                    threat = 5
            elif line.startswith("STRENGTHS:"):
                strengths = [s.strip() for s in line.replace("STRENGTHS:", "").split(";") if s.strip()]
            elif line.startswith("WEAKNESSES:"):
                weaknesses = [s.strip() for s in line.replace("WEAKNESSES:", "").split(";") if s.strip()]
            elif line.startswith("RECENT_MOVES:"):
                moves = [s.strip() for s in line.replace("RECENT_MOVES:", "").split(";") if s.strip()]
            elif line.startswith("OPPORTUNITIES:"):
                opportunities = [line.replace("OPPORTUNITIES:", "").strip()]
        
        return CompetitorProfile(
            competitor_id=competitor_id,
            name=metadata.get("name", competitor_id),
            last_updated=datetime.now(),
            business_model=metadata.get("model", "streaming"),
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            recent_moves=moves[:5],
            strategy_focus=strategy,
            threat_level=min(10, max(1, threat)),
            opportunity_areas=opportunities,
        )
    
    def build_all_profiles(
        self, 
        intel_items: List[ClassifiedIntel]
    ) -> Dict[str, CompetitorProfile]:
        """
        Build profiles for all competitors with sufficient intel.
        
        Returns:
            Dict of competitor_id -> CompetitorProfile
        """
        profiles = {}
        
        # Get unique competitors
        competitors = set(i.competitor.lower() for i in intel_items if i.competitor)
        
        for competitor_id in competitors:
            profile = self.build_profile(competitor_id, intel_items)
            if profile:
                profiles[competitor_id] = profile
        
        return profiles
    
    def summarize_landscape(self, profiles: Dict[str, CompetitorProfile]) -> str:
        """
        Create a summary of the competitive landscape.
        
        Returns:
            Markdown summary
        """
        if not profiles:
            return "No competitor profiles available."
        
        # Sort by threat level
        sorted_profiles = sorted(
            profiles.values(), 
            key=lambda p: p.threat_level, 
            reverse=True
        )
        
        summary = "## Competitive Landscape\n\n"
        
        # Threat summary
        high_threat = [p for p in sorted_profiles if p.threat_level >= 7]
        if high_threat:
            summary += "### High Threat Competitors\n"
            for p in high_threat:
                summary += f"- **{p.name}** (threat: {p.threat_level}/10): {p.strategy_focus}\n"
            summary += "\n"
        
        # Opportunity summary
        summary += "### Opportunity Areas\n"
        for p in sorted_profiles[:5]:
            if p.opportunity_areas:
                summary += f"- vs {p.name}: {p.opportunity_areas[0]}\n"
        
        return summary

