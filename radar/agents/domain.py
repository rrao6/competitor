"""
Domain Agents for Tubi Radar.

Specialized agents for different competitive intelligence perspectives:
- Product Agent: Platform, UX, devices, technology
- Content Agent: Shows, movies, content deals
- Marketing Agent: Campaigns, branding, partnerships
- AI/Ads Agent: Advertising technology, AI features, monetization
"""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from radar.agents.base import BaseAgent
from radar.config import get_config
from radar.schemas import DomainAnnotation, DomainAnnotationBatch
from radar.tools.db_tools import get_intel_for_domain, store_annotations_from_batch


class DomainAgent(BaseAgent):
    """
    Base class for domain-specific agents.
    
    Each domain agent analyzes intel from their specialized perspective
    and provides "so what" analysis, risk/opportunity assessment, and suggested actions.
    """
    
    # Override in subclasses
    agent_role = "domain_agent"
    domain_name = "General"
    category_filter: list[str] = []
    system_prompt = ""
    
    @property
    def temperature(self) -> float:
        """Use analysis temperature for domain reasoning."""
        if self._temperature_override is not None:
            return self._temperature_override
        return self.config.global_config.temperature.analysis
    
    def _build_intel_prompt(self, intel_items: list[dict]) -> str:
        """Build the prompt for intel analysis."""
        lines = [f"Analyze the following {len(intel_items)} intel items from your {self.domain_name} perspective:\n"]
        
        for item in intel_items:
            lines.append(f"---\n**Intel #{item['id']}**")
            lines.append(f"- Competitor: {item['competitor_id']}")
            lines.append(f"- Category: {item['category']}")
            lines.append(f"- Impact Score: {item['impact_score']}/10")
            lines.append(f"- Relevance Score: {item['relevance_score']}/10")
            lines.append(f"\nSummary: {item['summary']}")
            
            if item.get('entities'):
                lines.append(f"Entities: {', '.join(item['entities'])}")
            lines.append("")
        
        lines.append("---\nProvide your analysis for each intel item.")
        return "\n".join(lines)
    
    def _analyze_batch(self, intel_items: list[dict]) -> list[DomainAnnotation]:
        """
        Analyze a batch of intel items.
        
        Args:
            intel_items: List of intel dictionaries
        
        Returns:
            List of DomainAnnotation objects
        """
        if not intel_items:
            return []
        
        structured_llm = self.get_structured_llm(DomainAnnotationBatch)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_intel_prompt(intel_items)),
        ]
        
        try:
            result: DomainAnnotationBatch = structured_llm.invoke(messages)
            return result.annotations
        except Exception as e:
            print(f"[{self.agent_role}] Error analyzing batch: {e}")
            return []
    
    def run(
        self,
        run_id: int,
        min_relevance: Optional[float] = None,
        min_impact: Optional[float] = None,
    ) -> dict:
        """
        Execute the domain analysis.
        
        Args:
            run_id: The current run ID
            min_relevance: Minimum relevance score
            min_impact: Minimum impact score
        
        Returns:
            Dictionary with analysis results
        """
        config = get_config()
        
        min_relevance = min_relevance or config.global_config.min_relevance_score
        min_impact = min_impact or config.global_config.min_impact_score
        
        # Get filtered intel for this domain
        intel_items = get_intel_for_domain.invoke({
            "run_id": run_id,
            "category_filter": self.category_filter,
            "min_relevance": min_relevance,
            "min_impact": min_impact,
        })
        
        if not intel_items:
            print(f"[{self.agent_role}] No qualifying intel found")
            return {
                "analyzed": 0,
                "annotations_created": 0,
            }
        
        print(f"[{self.agent_role}] Analyzing {len(intel_items)} intel items...")
        
        # Analyze intel
        annotations = self._analyze_batch(intel_items)
        
        # Store annotations
        if annotations:
            stored_count = store_annotations_from_batch(annotations, self.agent_role)
            print(f"[{self.agent_role}] Created {stored_count} annotations")
        else:
            stored_count = 0
        
        return {
            "analyzed": len(intel_items),
            "annotations_created": stored_count,
        }


class ProductAgent(DomainAgent):
    """
    Product Intelligence Agent.
    
    Focuses on platform features, UX, apps, devices, and technology.
    """
    
    agent_role = "product_agent"
    domain_name = "Product & UX"
    category_filter = ["product", "ai_ads"]  # Also includes ad product
    
    system_prompt = """You are the Product Intelligence Agent for Tubi.

You only care about **product, UX, platform, and ad product** implications.

Tubi is a free ad-supported streaming service. You analyze competitive intel to understand how competitor product moves might affect Tubi's platform, apps, and user experience.

For each intel item:
1. Read the summarized intel and its category, impact, and relevance scores.
2. Ignore minor marketing fluff or generic industry commentary.

Provide for each item:

1. **so_what**: 2-3 sentences explaining why this matters (or not) to Tubi's product or ad experience. Be specific about implications.

2. **risk_or_opportunity**: Does this mainly represent a risk (competitor gaining advantage), an opportunity (chance for Tubi to differentiate or learn), or is it neutral?

3. **priority**: 
   - P0: Urgent/strategic - requires immediate attention or response
   - P1: Important but not urgent - should be on the roadmap or monitored
   - P2: Nice-to-know - interesting but low priority

4. **suggested_action**: Optional but valuable - a concrete next step (e.g., "Compare Roku's Live TV row to Tubi's home screen layout in a heuristic review within 2 weeks").

Focus on actionable insights that product and engineering teams can use."""


class ContentAgent(DomainAgent):
    """
    Content Intelligence Agent.
    
    Focuses on shows, movies, content deals, library changes, and originals.
    """
    
    agent_role = "content_agent"
    domain_name = "Content & Library"
    category_filter = ["content"]
    
    system_prompt = """You are the Content Intelligence Agent for Tubi.

You care about **content strategy, library composition, content deals, and originals**.

Tubi is a free ad-supported streaming service with a large library of movies and TV shows. You analyze competitive intel to understand how competitor content moves might affect Tubi's content positioning.

For each intel item:
1. Read the summarized intel about content deals, library changes, or original programming.
2. Consider how this affects the competitive landscape for AVOD/FAST content.

Provide for each item:

1. **so_what**: 2-3 sentences explaining the content strategy implications for Tubi. Consider exclusive deals, genre gaps, audience targeting.

2. **risk_or_opportunity**: 
   - Risk: Competitor acquiring content that would have been valuable for Tubi
   - Opportunity: Chance to acquire similar content or differentiate
   - Neutral: Standard industry moves with limited Tubi impact

3. **priority**:
   - P0: Major content moves that require immediate content team awareness
   - P1: Notable deals that should inform content strategy
   - P2: Routine content news

4. **suggested_action**: Optional - concrete content acquisition or programming suggestions.

Focus on insights the content and programming teams can action."""


class MarketingAgent(DomainAgent):
    """
    Marketing Intelligence Agent.
    
    Focuses on campaigns, branding, partnerships, and positioning.
    """
    
    agent_role = "marketing_agent"
    domain_name = "Marketing & Positioning"
    category_filter = ["marketing"]
    
    system_prompt = """You are the Marketing Intelligence Agent for Tubi.

You focus on **marketing campaigns, brand positioning, partnerships, and promotional strategies**.

Tubi is a free ad-supported streaming service. You analyze competitive intel to understand how competitors are positioning themselves and what marketing approaches are gaining traction.

For each intel item:
1. Read the summarized intel about marketing campaigns, partnerships, or brand moves.
2. Consider the messaging, target audience, and strategic positioning.

Provide for each item:

1. **so_what**: 2-3 sentences on what this means for Tubi's marketing and brand positioning. What message is the competitor sending? How might it affect viewer perception?

2. **risk_or_opportunity**:
   - Risk: Competitor messaging that could diminish Tubi's brand perception
   - Opportunity: Chance for Tubi to counter-position or learn from effective campaigns
   - Neutral: Standard marketing activity

3. **priority**:
   - P0: Major brand moves requiring marketing team response
   - P1: Notable campaigns worth monitoring or learning from
   - P2: Routine marketing news

4. **suggested_action**: Optional - specific marketing responses or campaign ideas.

Focus on brand and marketing insights that inform Tubi's go-to-market strategy."""


class AIAdsAgent(DomainAgent):
    """
    AI & Advertising Intelligence Agent.
    
    Focuses on advertising technology, AI features, targeting, and monetization.
    """
    
    agent_role = "ai_ads_agent"
    domain_name = "AI & Ads / Pricing"
    category_filter = ["ai_ads", "pricing"]
    
    system_prompt = """You are the AI & Advertising Intelligence Agent for Tubi.

You focus on **advertising technology, AI/ML features, ad products, targeting capabilities, and pricing/monetization**.

Tubi is a free ad-supported streaming service where advertising is the primary revenue driver. You analyze competitive intel about ad tech innovation and pricing strategies.

For each intel item:
1. Read the summarized intel about ad technology, AI features, or pricing changes.
2. Consider the implications for Tubi's ad business and technology roadmap.

Provide for each item:

1. **so_what**: 2-3 sentences on the ad tech or monetization implications. How might this affect advertiser preference? What capabilities are competitors building?

2. **risk_or_opportunity**:
   - Risk: Competitor ad tech that could attract advertisers away from Tubi
   - Opportunity: Technology Tubi should consider or gaps to exploit
   - Neutral: Standard industry evolution

3. **priority**:
   - P0: Major ad tech moves requiring immediate ad product team attention
   - P1: Notable capabilities worth building or countering
   - P2: Routine ad industry news

4. **suggested_action**: Optional - specific ad product or technology recommendations.

Focus on insights that inform Tubi's ad product strategy and technology investments."""


def run_all_domain_agents(run_id: int) -> dict:
    """
    Run all domain agents and return combined results.
    
    Args:
        run_id: The current run ID
    
    Returns:
        Combined results from all domain agents
    """
    results = {}
    
    agents = [
        ProductAgent(),
        ContentAgent(),
        MarketingAgent(),
        AIAdsAgent(),
    ]
    
    for agent in agents:
        print(f"\n[{agent.agent_role}] Starting analysis...")
        results[agent.agent_role] = agent.run(run_id=run_id)
    
    return results

