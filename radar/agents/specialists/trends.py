"""
Trend Tracker for Tubi Radar.

Compares to historical data, identifies emerging patterns,
and predicts where the industry is heading.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_settings
from radar.agents.classifier_swarm import ClassifiedIntel


@dataclass
class Trend:
    """An identified industry trend."""
    trend_id: str
    name: str
    category: str  # technology, content, business, advertising, consumer
    direction: str  # accelerating, stable, declining, emerging
    strength: int  # 1-10
    description: str
    evidence: List[str]
    prediction: str
    confidence: float
    timeframe: str  # 3-month, 6-month, 12-month


class TrendTracker:
    """
    Tracks and predicts industry trends based on intel patterns.
    
    Capabilities:
    - Pattern recognition across multiple intel items
    - Historical comparison (last 7/30/90 days when data available)
    - Trend prediction and confidence scoring
    """
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
    
    def analyze(
        self, 
        intel_items: List[ClassifiedIntel],
        historical_context: Optional[str] = None
    ) -> List[Trend]:
        """
        Identify trends from intel items.
        
        Args:
            intel_items: Current classified intel
            historical_context: Optional summary of past intel
            
        Returns:
            List of identified trends
        """
        if not intel_items:
            return []
        
        # Group intel by category
        by_category: Dict[str, List[ClassifiedIntel]] = {}
        for item in intel_items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)
        
        # Build prompt with category distribution
        intel_text = ""
        for i, item in enumerate(intel_items[:30]):
            intel_text += f"\n{i+1}. [{item.competitor}|{item.category}] {item.summary}"
        
        category_summary = ", ".join([f"{k}: {len(v)}" for k, v in by_category.items()])
        
        historical_section = ""
        if historical_context:
            historical_section = f"\n\nHistorical context (previous period):\n{historical_context}"
        
        prompt = f"""You are a streaming industry trend analyst for Tubi.

Analyze this intel and identify TRENDS (patterns across multiple items):

Intel distribution by category: {category_summary}
{intel_text}
{historical_section}

For each TREND you identify, output one line:
ID|CATEGORY|DIRECTION|STRENGTH|NAME|DESCRIPTION|PREDICTION|TIMEFRAME

- ID: Unique trend ID (e.g., T1, T2)
- CATEGORY: technology/content/business/advertising/consumer
- DIRECTION: accelerating/stable/declining/emerging
- STRENGTH: 1-10 (how strong is this trend)
- NAME: Short trend name
- DESCRIPTION: What is happening
- PREDICTION: Where this is heading
- TIMEFRAME: 3-month/6-month/12-month

Identify 5-10 trends. Focus on streaming, AVOD, FAST, CTV. Output (no headers):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content)
        except Exception as e:
            print(f"        TrendTracker error: {e}")
            return []
    
    def _parse_response(self, text: str) -> List[Trend]:
        """Parse LLM response into Trend objects."""
        trends = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 8:
                continue
            
            try:
                trend_id = parts[0].strip()
                category = parts[1].strip().lower()
                direction = parts[2].strip().lower()
                strength = int(parts[3].strip())
                name = parts[4].strip()
                description = parts[5].strip()
                prediction = parts[6].strip()
                timeframe = parts[7].strip().lower()
                
                valid_categories = ["technology", "content", "business", "advertising", "consumer"]
                if category not in valid_categories:
                    category = "business"
                
                valid_directions = ["accelerating", "stable", "declining", "emerging"]
                if direction not in valid_directions:
                    direction = "stable"
                
                valid_timeframes = ["3-month", "6-month", "12-month"]
                if timeframe not in valid_timeframes:
                    timeframe = "6-month"
                
                # Calculate confidence based on strength and direction
                confidence = 0.7
                if direction == "accelerating":
                    confidence = 0.85
                elif direction == "emerging":
                    confidence = 0.6
                
                trends.append(Trend(
                    trend_id=trend_id,
                    name=name,
                    category=category,
                    direction=direction,
                    strength=min(10, max(1, strength)),
                    description=description,
                    evidence=[],  # Could be populated with matching intel IDs
                    prediction=prediction,
                    confidence=confidence,
                    timeframe=timeframe,
                ))
            except Exception:
                continue
        
        # Sort by strength
        trends.sort(key=lambda x: x.strength, reverse=True)
        
        return trends
    
    def compare_periods(
        self,
        current_trends: List[Trend],
        previous_trends: List[Trend]
    ) -> Dict[str, Any]:
        """
        Compare trends between two periods.
        
        Returns:
            Dict with new trends, accelerating trends, declining trends
        """
        current_names = {t.name.lower() for t in current_trends}
        previous_names = {t.name.lower() for t in previous_trends}
        
        new_trends = [t for t in current_trends if t.name.lower() not in previous_names]
        disappeared = [t for t in previous_trends if t.name.lower() not in current_names]
        
        return {
            "new_trends": new_trends,
            "disappeared_trends": disappeared,
            "trend_count_change": len(current_trends) - len(previous_trends),
        }

