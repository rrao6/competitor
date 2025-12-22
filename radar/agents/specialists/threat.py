"""
Threat Analyst for Tubi Radar.

Identifies competitive threats to Tubi, scores severity,
and suggests defensive actions.
"""
from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_settings
from radar.agents.classifier_swarm import ClassifiedIntel


@dataclass
class ThreatAssessment:
    """A competitive threat assessment."""
    intel_id: int
    competitor: str
    threat_type: str  # direct, indirect, potential, existential
    severity: int  # 1-10
    description: str
    defensive_action: str
    timeframe: str  # immediate, short-term, long-term
    confidence: float  # 0-1


class ThreatAnalyst:
    """
    Analyzes intel for competitive threats to Tubi.
    
    Focus areas:
    - Direct threats (competitors copying Tubi's model)
    - Indirect threats (market shifts that hurt AVOD)
    - Existential threats (industry changes that could eliminate AVOD)
    """
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
    
    def analyze(self, intel_items: List[ClassifiedIntel]) -> List[ThreatAssessment]:
        """
        Analyze intel for threats to Tubi.
        
        Args:
            intel_items: Classified intel from the classifier swarm
            
        Returns:
            List of threat assessments
        """
        if not intel_items:
            return []
        
        # Focus on high-impact items
        high_impact = [i for i in intel_items if i.impact >= 5 or i.relevance >= 5]
        
        if not high_impact:
            high_impact = intel_items[:20]
        
        # Build prompt
        intel_text = ""
        for i, item in enumerate(high_impact[:25]):
            intel_text += f"\n{i+1}. [{item.competitor}] {item.summary}"
        
        prompt = f"""You are a competitive threat analyst for Tubi, the #1 free streaming service (AVOD/FAST).

Analyze these intel items and identify threats to Tubi's business:

{intel_text}

For each THREAT (not opportunity), output one line:
NUM|TYPE|SEVERITY|TIMEFRAME|THREAT|ACTION

- NUM: Intel number
- TYPE: direct/indirect/potential/existential
- SEVERITY: 1-10 (10 = critical threat)
- TIMEFRAME: immediate/short-term/long-term
- THREAT: What specifically threatens Tubi
- ACTION: Recommended defensive action

Only include items that represent genuine threats. Skip positive news or neutral updates.
Output (no headers):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, high_impact)
        except Exception as e:
            print(f"        ThreatAnalyst error: {e}")
            return []
    
    def _parse_response(self, text: str, intel_items: List[ClassifiedIntel]) -> List[ThreatAssessment]:
        """Parse LLM response into ThreatAssessment objects."""
        assessments = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 6:
                continue
            
            try:
                num = int(parts[0].strip().replace(".", "")) - 1
                if num < 0 or num >= len(intel_items):
                    continue
                
                intel = intel_items[num]
                threat_type = parts[1].strip().lower()
                severity = int(parts[2].strip())
                timeframe = parts[3].strip().lower()
                description = parts[4].strip()
                action = parts[5].strip()
                
                valid_types = ["direct", "indirect", "potential", "existential"]
                if threat_type not in valid_types:
                    threat_type = "potential"
                
                valid_timeframes = ["immediate", "short-term", "long-term"]
                if timeframe not in valid_timeframes:
                    timeframe = "short-term"
                
                assessments.append(ThreatAssessment(
                    intel_id=intel.article_id,
                    competitor=intel.competitor,
                    threat_type=threat_type,
                    severity=min(10, max(1, severity)),
                    description=description,
                    defensive_action=action,
                    timeframe=timeframe,
                    confidence=0.8,
                ))
            except Exception:
                continue
        
        # Sort by severity
        assessments.sort(key=lambda x: x.severity, reverse=True)
        
        return assessments

