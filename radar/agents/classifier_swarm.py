"""
Parallel Classifier Swarm for Tubi Radar.

4 parallel classifier workers processing articles simultaneously.
Uses gpt-4o-mini for speed, with results merged and deduplicated.
Includes intelligent grouping of similar articles into themed intel.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_config, get_settings


@dataclass
class ClassifiedIntel:
    """A classified piece of intel."""
    article_id: int
    competitor: str
    title: str
    url: str
    summary: str
    category: str
    impact: float
    relevance: float
    entities: List[str]
    # Grouped articles - when multiple articles cover same story
    related_urls: List[str] = field(default_factory=list)
    source_count: int = 1
    
    @property
    def hash(self) -> str:
        """Unique hash for deduplication."""
        return hashlib.md5(f"{self.title}{self.url}".encode()).hexdigest()
    
    @property
    def theme_hash(self) -> str:
        """Hash based on summary for grouping similar stories."""
        # Normalize summary for comparison
        normalized = re.sub(r'[^\w\s]', '', self.summary.lower())
        words = sorted(normalized.split())[:10]  # First 10 words sorted
        return hashlib.md5(' '.join(words).encode()).hexdigest()


class ClassifierWorker:
    """A single classifier worker."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
    
    def classify_batch(self, articles: List[dict]) -> List[ClassifiedIntel]:
        """Classify a batch of articles."""
        if not articles:
            return []
        
        # Build prompt
        article_text = ""
        for i, a in enumerate(articles):
            title = (a.get('title') or '')[:120]
            snippet = (a.get('raw_snippet') or '')[:400]
            article_text += f"\n{i+1}. [{a.get('competitor_id', 'unknown')}] {title}\n   {snippet}\n"
        
        prompt = f"""Extract key facts from streaming industry articles.

Output format - ONE LINE per article:
NUM|CATEGORY|IMPACT|RELEVANCE|ENTITIES|SUMMARY

CATEGORY (pick one):
- strategic: M&A, partnerships, subscriber/revenue numbers, market share
- product: New features, app updates, platform changes
- content: Shows, movies, licensing, originals announcements
- marketing: Ad campaigns, brand partnerships
- ai_ads: Ad tech, CTV targeting, programmatic
- pricing: Price changes, new tiers, bundles

SCORES:
- IMPACT: 5-10 (5=minor, 7=notable, 9=major news)
- RELEVANCE: 5-10 (streaming industry relevance)

ENTITIES: Company names only (comma-separated)

SUMMARY RULES - CRITICAL:
Write exactly what happened. Include WHO, WHAT, and specific NUMBERS.

CORRECT FORMAT:
"Tubi reached 100M monthly active users in June 2025"
"Netflix acquired Warner Bros for $82.7 billion"
"Roku added 40 new FAST channels in UK"
"Amazon shutting down Freevee, moving content to Prime Video"
"Disney+ reached 150M subscribers, up 12% YoY"

FORBIDDEN WORDS (never use):
- "highlighting" "indicating" "underscoring" "suggesting"
- "significant" "notable" "important" "key" "major" (without numbers)
- "amid" "landscape" "trajectory" "evolution"
- "competitive advantage" "market position" "growth trajectory"
- "could impact" "may affect" "aims to"

If you can't state a specific fact with numbers, SKIP the article.

Articles:
{article_text}

Output:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, articles)
        except Exception as e:
            print(f"        Worker {self.worker_id} error: {e}")
            return []
    
    def _parse_response(self, text: str, articles: List[dict]) -> List[ClassifiedIntel]:
        """Parse LLM response into ClassifiedIntel objects."""
        items = []
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 6:
                continue
            
            try:
                num = int(parts[0].strip().replace(".", "")) - 1
                if num < 0 or num >= len(articles):
                    continue
                
                article = articles[num]
                category = parts[1].strip().lower()
                impact = float(parts[2].strip())
                relevance = float(parts[3].strip())
                entities = [e.strip() for e in parts[4].split(",") if e.strip()]
                summary = "|".join(parts[5:]).strip()
                
                valid_categories = ["strategic", "product", "content", "marketing", "ai_ads", "pricing"]
                if category not in valid_categories:
                    category = "strategic"
                
                items.append(ClassifiedIntel(
                    article_id=article.get('id', 0),
                    competitor=article.get('competitor_id', 'unknown'),
                    title=article.get('title', ''),
                    url=article.get('url', ''),
                    summary=summary,
                    category=category,
                    impact=min(10, max(1, impact)),
                    relevance=min(10, max(1, relevance)),
                    entities=entities,
                ))
            except Exception:
                continue
        
        return items


class ClassifierSwarm:
    """
    Swarm of 4 parallel classifiers for maximum throughput.
    
    Each worker processes ~50 articles simultaneously.
    Results are merged and deduplicated.
    """
    
    def __init__(self, num_workers: int = 4, batch_size: int = 50):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.workers = [ClassifierWorker(i) for i in range(num_workers)]
    
    def classify_all(self, articles: List[dict]) -> List[ClassifiedIntel]:
        """
        Classify all articles using parallel workers.
        
        Args:
            articles: List of article dicts
            
        Returns:
            Deduplicated list of ClassifiedIntel
        """
        if not articles:
            return []
        
        # Split into batches
        batches = []
        for i in range(0, len(articles), self.batch_size):
            batches.append(articles[i:i + self.batch_size])
        
        all_items = []
        seen_hashes = set()
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Distribute batches across workers
            futures = {}
            for i, batch in enumerate(batches):
                worker = self.workers[i % self.num_workers]
                future = executor.submit(worker.classify_batch, batch)
                futures[future] = i
            
            # Collect results
            for future in as_completed(futures):
                try:
                    items = future.result()
                    for item in items:
                        # Deduplicate
                        if item.hash not in seen_hashes:
                            seen_hashes.add(item.hash)
                            all_items.append(item)
                except Exception as e:
                    print(f"        Batch error: {e}")
        
        # Sort by impact
        all_items.sort(key=lambda x: (x.impact, x.relevance), reverse=True)
        
        return all_items


def run_classifier_swarm(articles: List[dict]) -> List[ClassifiedIntel]:
    """
    Convenience function to run the classifier swarm.
    
    Args:
        articles: List of article dicts
        
    Returns:
        List of ClassifiedIntel (grouped by theme)
    """
    swarm = ClassifierSwarm()
    raw_intel = swarm.classify_all(articles)
    
    # Group similar stories together
    grouped_intel = group_similar_intel(raw_intel)
    
    return grouped_intel


def group_similar_intel(intel_items: List[ClassifiedIntel]) -> List[ClassifiedIntel]:
    """
    Group similar intel items into consolidated themes.
    
    Articles about the same story are merged into one intel item
    with multiple source URLs.
    
    Args:
        intel_items: Raw classified intel
        
    Returns:
        Grouped/deduplicated intel
    """
    if not intel_items:
        return []
    
    # Group by similarity
    groups: Dict[str, List[ClassifiedIntel]] = {}
    
    for item in intel_items:
        # Create grouping key based on entities + category
        # This catches "Netflix acquires Warner Bros" from multiple sources
        key_entities = sorted([e.lower() for e in item.entities[:3]])
        group_key = f"{item.category}|{'|'.join(key_entities)}"
        
        # Also check for similar summaries within same category
        found_group = None
        for existing_key, existing_items in groups.items():
            if existing_key.startswith(item.category):
                # Check summary similarity
                if _are_summaries_similar(item.summary, existing_items[0].summary):
                    found_group = existing_key
                    break
        
        if found_group:
            groups[found_group].append(item)
        elif group_key in groups:
            groups[group_key].append(item)
        else:
            groups[group_key] = [item]
    
    # Merge groups into single intel items
    merged_intel = []
    
    for group_key, items in groups.items():
        if len(items) == 1:
            merged_intel.append(items[0])
        else:
            # Merge multiple items into one
            merged = _merge_intel_group(items)
            merged_intel.append(merged)
    
    # Sort by impact
    merged_intel.sort(key=lambda x: (x.impact, x.relevance), reverse=True)
    
    return merged_intel


def _are_summaries_similar(summary1: str, summary2: str, threshold: float = 0.7) -> bool:
    """
    Check if two summaries are about the SAME story (not just similar topic).
    
    We want to group:
    - "Netflix acquires Warner Bros for $82B" from source A
    - "Netflix acquires Warner Bros for $82B" from source B
    
    We do NOT want to group:
    - "Netflix acquires Warner Bros for $82B" 
    - "Netflix acquires Warner Bros for $72B" (different amount = different story)
    - "Warner Bros prefers Netflix over Paramount" (different angle)
    """
    # Normalize
    def normalize(s):
        return re.sub(r'[^\w\s]', '', s.lower())
    
    norm1 = normalize(summary1)
    norm2 = normalize(summary2)
    
    # Extract key numbers (prices, percentages, years)
    numbers1 = set(re.findall(r'\d+(?:\.\d+)?', summary1))
    numbers2 = set(re.findall(r'\d+(?:\.\d+)?', summary2))
    
    # If both have numbers and they differ significantly, NOT same story
    if numbers1 and numbers2 and numbers1 != numbers2:
        # Check if any number differs by more than 10%
        for n1 in numbers1:
            for n2 in numbers2:
                try:
                    v1, v2 = float(n1), float(n2)
                    if v1 > 0 and v2 > 0:
                        diff = abs(v1 - v2) / max(v1, v2)
                        if diff > 0.1:  # More than 10% difference
                            return False
                except ValueError:
                    pass
    
    # Get significant words (excluding stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
                 'its', 'their', 'it', 'they', 'new', 'says', 'said', 'reports', 'according'}
    
    words1 = set(norm1.split()) - stopwords
    words2 = set(norm2.split()) - stopwords
    
    if not words1 or not words2:
        return False
    
    # Check for key differentiating words
    # If one mentions "prefers" or "over" or "vs" and other doesn't, different stories
    comparison_words = {'prefers', 'over', 'vs', 'versus', 'compared', 'instead', 'rather', 'chooses'}
    has_comparison1 = bool(words1 & comparison_words)
    has_comparison2 = bool(words2 & comparison_words)
    if has_comparison1 != has_comparison2:
        return False
    
    # Jaccard similarity with higher threshold
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    similarity = intersection / union if union > 0 else 0
    
    # Also check that they share at least 60% of significant content words
    min_shared = min(len(words1), len(words2)) * 0.6
    
    return similarity >= threshold and intersection >= min_shared


def _merge_intel_group(items: List[ClassifiedIntel]) -> ClassifiedIntel:
    """Merge a group of similar intel items into one."""
    # Use the highest impact item as the base
    items.sort(key=lambda x: (x.impact, x.relevance), reverse=True)
    base = items[0]
    
    # Collect all URLs
    all_urls = [item.url for item in items]
    
    # Combine entities
    all_entities = set()
    for item in items:
        all_entities.update(item.entities)
    
    # Average the scores
    avg_impact = sum(item.impact for item in items) / len(items)
    avg_relevance = sum(item.relevance for item in items) / len(items)
    
    # Use max impact/relevance since multiple sources = more important
    max_impact = max(item.impact for item in items)
    max_relevance = max(item.relevance for item in items)
    
    # Create merged summary
    summary = base.summary
    if len(items) > 1:
        summary = f"[{len(items)} sources] {base.summary}"
    
    return ClassifiedIntel(
        article_id=base.article_id,
        competitor=base.competitor,
        title=base.title,
        url=base.url,
        summary=summary,
        category=base.category,
        impact=max_impact,  # Use max since multi-source = important
        relevance=max_relevance,
        entities=list(all_entities)[:10],
        related_urls=all_urls[1:],  # All other URLs
        source_count=len(items),
    )

