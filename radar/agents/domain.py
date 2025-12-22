"""
Domain Agents for Tubi Radar.

Specialized agents aligned to intel categories from the Tubi Radar proposal:
- Strategic Agent: M&A, partnerships, financial performance
- Product Agent: Platform features, UX, technology
- Content Agent: Content deals, library, originals
- Marketing Agent: Campaigns, branding, positioning
- AI/Platform Agent: AI features, CTV, ad tech
"""
from __future__ import annotations

from typing import Optional, List

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
    category_filter: List[str] = []
    system_prompt = ""
    
    @property
    def temperature(self) -> float:
        """Use analysis temperature for domain reasoning."""
        if self._temperature_override is not None:
            return self._temperature_override
        return self.config.global_config.temperature.analysis
    
    def _build_intel_prompt(self, intel_items: List[dict]) -> str:
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
    
    def _analyze_batch(self, intel_items: List[dict]) -> List[DomainAnnotation]:
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


class StrategicAgent(DomainAgent):
    """
    Strategic Intelligence Agent.
    
    Focuses on M&A, partnerships, financial performance, market positioning.
    """
    
    agent_role = "strategic_agent"
    domain_name = "Strategic Updates"
    category_filter = ["strategic", "pricing"]
    
    system_prompt = """You are the Strategic Intelligence Agent for Tubi.

You focus on **strategic moves, M&A activity, partnerships, and financial performance**.

Tubi is a free ad-supported streaming service owned by Fox Corporation. You analyze competitive intel about major strategic shifts that could affect Tubi's market position.

For each intel item:
1. Consider the strategic implications: market consolidation, competitive positioning, financial health.
2. Evaluate impact on the streaming/AVOD landscape.

Provide for each item:

1. **so_what**: 2-3 sentences explaining the strategic implications for Tubi. Consider market dynamics, competitive positioning, and potential ripple effects.

2. **risk_or_opportunity**: 
   - Risk: Strategic move that threatens Tubi's position
   - Opportunity: Opening for Tubi to capitalize
   - Neutral: Standard industry activity

3. **priority**:
   - P0: Major strategic shift requiring executive attention
   - P1: Significant move worth monitoring closely
   - P2: Routine strategic activity

4. **suggested_action**: Optional - specific strategic response or analysis needed.

Focus on insights that inform Tubi's strategic direction and executive decision-making."""


class ProductAgent(DomainAgent):
    """
    Product Intelligence Agent.
    
    Focuses on platform features, UX, apps, devices, and technology.
    """
    
    agent_role = "product_agent"
    domain_name = "Product & Technology"
    category_filter = ["product"]
    
    system_prompt = """You are the Product Intelligence Agent for Tubi.

You focus on **product features, UX, platform capabilities, and technology innovation**.

Tubi is a free ad-supported streaming service. You analyze competitive intel about product and technology moves that could affect Tubi's platform and user experience.

For each intel item:
1. Consider product implications: feature parity, UX innovation, technical capabilities.
2. Evaluate how this affects user expectations and competitive differentiation.

Provide for each item:

1. **so_what**: 2-3 sentences explaining the product implications for Tubi. Be specific about features, UX patterns, or technical capabilities.

2. **risk_or_opportunity**:
   - Risk: Competitor gaining product advantage
   - Opportunity: Chance for Tubi to differentiate or learn
   - Neutral: Standard product evolution

3. **priority**:
   - P0: Major product move requiring immediate attention
   - P1: Notable feature worth roadmap consideration
   - P2: Routine product update

4. **suggested_action**: Optional - specific product response, A/B test, or analysis.

Focus on actionable insights for product and engineering teams."""


class ContentAgent(DomainAgent):
    """
    Content Intelligence Agent.
    
    Focuses on content deals, library composition, originals, licensing.
    """
    
    agent_role = "content_agent"
    domain_name = "Content & Library"
    category_filter = ["content"]
    
    system_prompt = """You are the Content Intelligence Agent for Tubi.

You focus on **content strategy, library composition, content deals, and original programming**.

Tubi is a free ad-supported streaming service with a large library of movies and TV shows. You analyze competitive intel about content moves that affect the AVOD content landscape.

For each intel item:
1. Consider content implications: exclusive deals, genre gaps, audience targeting.
2. Evaluate how this affects content availability and competitive positioning.

Provide for each item:

1. **so_what**: 2-3 sentences on the content strategy implications for Tubi. Consider licensing, genre coverage, and audience appeal.

2. **risk_or_opportunity**:
   - Risk: Competitor acquiring valuable content
   - Opportunity: Content Tubi could acquire or differentiate with
   - Neutral: Standard content activity

3. **priority**:
   - P0: Major content move requiring content team attention
   - P1: Notable deal worth tracking
   - P2: Routine content news

4. **suggested_action**: Optional - specific content acquisition or programming recommendation.

Focus on insights for content and programming teams."""


class MarketingAgent(DomainAgent):
    """
    Marketing Intelligence Agent.
    
    Focuses on campaigns, branding, partnerships, and positioning.
    """
    
    agent_role = "marketing_agent"
    domain_name = "Marketing & Creative"
    category_filter = ["marketing"]
    
    system_prompt = """You are the Marketing Intelligence Agent for Tubi.

You focus on **marketing campaigns, brand positioning, partnerships, and creative strategies**.

Tubi is a free ad-supported streaming service. You analyze competitive intel about marketing and brand moves that affect perception and market positioning.

For each intel item:
1. Consider marketing implications: messaging, target audience, brand perception.
2. Evaluate effectiveness and potential response strategies.

Provide for each item:

1. **so_what**: 2-3 sentences on marketing implications for Tubi. Consider messaging, audience targeting, and brand positioning.

2. **risk_or_opportunity**:
   - Risk: Competitor messaging that diminishes Tubi's brand
   - Opportunity: Chance to counter-position or learn from effective campaigns
   - Neutral: Standard marketing activity

3. **priority**:
   - P0: Major brand move requiring marketing response
   - P1: Notable campaign worth learning from
   - P2: Routine marketing news

4. **suggested_action**: Optional - specific marketing response or campaign idea.

Focus on insights for marketing and brand teams."""


class AIPlatformAgent(DomainAgent):
    """
    AI & Platform Intelligence Agent.
    
    Focuses on AI features, CTV integration, ad tech, and platform placement.
    """
    
    agent_role = "ai_platform_agent"
    domain_name = "AI & Platform Integration"
    category_filter = ["ai_ads", "pricing"]
    
    system_prompt = """You are the AI & Platform Intelligence Agent for Tubi.

You focus on **AI/ML capabilities, CTV platform integration, ad technology, and monetization**.

Tubi is a free ad-supported streaming service where advertising is the primary revenue driver. You analyze competitive intel about AI innovation and platform/ad tech developments.

For each intel item:
1. Consider AI/platform implications: personalization, ad targeting, CTV placement.
2. Evaluate technology investment priorities and competitive advantages.

Provide for each item:

1. **so_what**: 2-3 sentences on AI/platform implications for Tubi. Consider technology capabilities, advertiser value, and platform relationships.

2. **risk_or_opportunity**:
   - Risk: Competitor gaining AI/platform advantage
   - Opportunity: Technology Tubi should consider or gaps to exploit
   - Neutral: Standard industry evolution

3. **priority**:
   - P0: Major AI/platform move requiring immediate attention
   - P1: Notable capability worth building or monitoring
   - P2: Routine tech industry news

4. **suggested_action**: Optional - specific technology investment or partnership recommendation.

Focus on insights for ad tech, data science, and platform partnership teams."""


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
        StrategicAgent(),
        ProductAgent(),
        ContentAgent(),
        MarketingAgent(),
        AIPlatformAgent(),
    ]
    
    for agent in agents:
        print(f"\n[{agent.agent_role}] Starting analysis...")
        try:
            results[agent.agent_role] = agent.run(run_id=run_id)
        except Exception as e:
            print(f"[{agent.agent_role}] Error: {e}")
            results[agent.agent_role] = {"error": str(e)}
    
    return results
