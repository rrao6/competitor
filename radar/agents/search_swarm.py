"""
Parallel Web Search Swarm for Tubi Radar.

5 specialized search agents, each with different query strategies:
- Agent 1: Breaking news
- Agent 2: Deals & M&A
- Agent 3: Product launches
- Agent 4: Earnings & metrics
- Agent 5: Industry trends
"""
from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib

from openai import OpenAI

from radar.config import get_config, get_settings
from radar.tools.rss import ArticleCandidate


@dataclass
class SearchStrategy:
    """A search strategy with query patterns."""
    name: str
    query_templates: List[str]
    focus_keywords: List[str]


# Define the 5 search strategies
STRATEGIES = [
    SearchStrategy(
        name="breaking_news",
        query_templates=[
            "{competitor} announces today",
            "{competitor} streaming news December 2025",
            "{competitor} breaking news streaming",
        ],
        focus_keywords=["announces", "breaking", "just", "today", "launches"]
    ),
    SearchStrategy(
        name="deals_ma",
        query_templates=[
            "streaming acquisition deal 2025",
            "{competitor} partnership deal",
            "streaming merger acquisition news",
            "FAST channel acquisition",
        ],
        focus_keywords=["acquisition", "merger", "deal", "partnership", "investment"]
    ),
    SearchStrategy(
        name="product_launches",
        query_templates=[
            "{competitor} new feature launch",
            "streaming app update 2025",
            "{competitor} streaming technology",
            "AVOD FAST new features",
        ],
        focus_keywords=["launch", "release", "update", "feature", "app", "technology"]
    ),
    SearchStrategy(
        name="earnings_metrics",
        query_templates=[
            "streaming subscribers Q4 2025",
            "{competitor} subscriber growth",
            "streaming earnings report 2025",
            "AVOD revenue growth",
        ],
        focus_keywords=["subscribers", "earnings", "revenue", "growth", "quarterly", "report"]
    ),
    SearchStrategy(
        name="industry_trends",
        query_templates=[
            "AVOD FAST growth 2025",
            "connected TV advertising trends",
            "streaming industry forecast",
            "cord cutting trends 2025",
            "streaming bundling trends",
        ],
        focus_keywords=["trend", "forecast", "growth", "future", "industry"]
    ),
]

# Key competitors to search for
COMPETITORS = [
    "Netflix", "Disney Plus", "Hulu", "Amazon Prime Video", "Peacock",
    "Max HBO", "Paramount Plus", "Pluto TV", "Roku", "YouTube TV",
    "Freevee", "Apple TV", "Xumo", "Sling TV", "Crunchyroll"
]


class SearchAgent:
    """A specialized search agent with a specific strategy."""
    
    def __init__(self, strategy: SearchStrategy, agent_id: int):
        self.strategy = strategy
        self.agent_id = agent_id
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    
    def search(self, max_results_per_query: int = 5) -> List[ArticleCandidate]:
        """Execute searches using this agent's strategy."""
        if not self.client:
            return []
        
        results = []
        
        # Generate queries from templates
        queries = self._generate_queries()
        
        for query in queries[:8]:  # Limit queries per agent
            try:
                search_results = self._execute_search(query, max_results_per_query)
                results.extend(search_results)
            except Exception as e:
                print(f"        Search agent {self.agent_id} error on '{query}': {e}")
        
        return results
    
    def _generate_queries(self) -> List[str]:
        """Generate search queries from templates."""
        queries = []
        
        for template in self.strategy.query_templates:
            if "{competitor}" in template:
                # Generate for each competitor
                for competitor in COMPETITORS[:5]:  # Top 5 competitors
                    queries.append(template.format(competitor=competitor))
            else:
                queries.append(template)
        
        return queries
    
    def _execute_search(self, query: str, max_results: int) -> List[ArticleCandidate]:
        """Execute a single search query."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-search-preview",
                web_search_options={"search_context_size": "medium"},
                messages=[{
                    "role": "user",
                    "content": f"""Search for the latest news about: {query}

Return exactly {max_results} recent news items in this EXACT format (one per line):

TITLE | URL | SNIPPET

- TITLE: Article headline
- URL: Full article URL
- SNIPPET: 2-3 sentence summary of the article

Only include news from the last 7 days. Focus on streaming, AVOD, FAST, CTV news.
Output (no headers, one per line):"""
                }],
                max_tokens=1500
            )
            
            return self._parse_results(response.choices[0].message.content, query)
        except Exception as e:
            return []
    
    def _parse_results(self, text: str, query: str) -> List[ArticleCandidate]:
        """Parse search results into ArticleCandidate objects."""
        results = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 3:
                continue
            
            try:
                title = parts[0].strip()[:200]
                url = parts[1].strip()
                snippet = "|".join(parts[2:]).strip()[:500]
                
                if not url.startswith("http"):
                    continue
                
                # Create hash
                content_hash = hashlib.md5(f"{title}{url}".encode()).hexdigest()
                
                results.append(ArticleCandidate(
                    title=title,
                    url=url,
                    published=datetime.now(),
                    raw_snippet=snippet,
                    source=f"web_search_{self.strategy.name}",
                    competitor_id="industry",
                    hash=content_hash,
                ))
            except Exception:
                continue
        
        return results


class SearchSwarm:
    """
    Swarm of 5 parallel search agents with different strategies.
    
    Each agent focuses on a specific type of news:
    1. Breaking news
    2. Deals & M&A
    3. Product launches
    4. Earnings & metrics
    5. Industry trends
    """
    
    def __init__(self):
        self.agents = [
            SearchAgent(strategy, i) 
            for i, strategy in enumerate(STRATEGIES)
        ]
    
    def search_all(self, max_results_per_query: int = 5) -> List[ArticleCandidate]:
        """
        Execute all search strategies in parallel.
        
        Returns:
            Deduplicated list of ArticleCandidate
        """
        all_results = []
        seen_urls = set()
        
        # Run all agents in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(agent.search, max_results_per_query): agent
                for agent in self.agents
            }
            
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    results = future.result()
                    for r in results:
                        # Deduplicate by URL
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            all_results.append(r)
                except Exception as e:
                    print(f"        Search agent {agent.agent_id} failed: {e}")
        
        return all_results


def run_search_swarm(max_results_per_query: int = 5) -> List[ArticleCandidate]:
    """
    Convenience function to run the search swarm.
    
    Returns:
        List of ArticleCandidate from all search strategies
    """
    swarm = SearchSwarm()
    return swarm.search_all(max_results_per_query)

