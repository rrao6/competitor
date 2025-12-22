"""
Real-Time Streaming Engine for Tubi Radar.

Provides:
- Continuous RSS polling (configurable interval)
- Instant web search on breaking news keywords
- Priority alerts for high-impact intel
- Dashboard live updates support
"""
from __future__ import annotations

import time
import threading
import queue
from typing import List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib

from radar.config import get_config
from radar.tools.rss import fetch_all_feeds_parallel, ArticleCandidate
from radar.agents.classifier_swarm import run_classifier_swarm, ClassifiedIntel
from radar.agents.search_swarm import SearchSwarm


@dataclass
class StreamEvent:
    """An event from the streaming engine."""
    event_type: str  # new_intel, alert, status, error
    timestamp: datetime
    data: Any
    priority: int = 0  # 0=normal, 1=high, 2=critical


@dataclass
class StreamConfig:
    """Configuration for the streaming engine."""
    poll_interval_seconds: int = 900  # 15 minutes
    web_search_interval_seconds: int = 3600  # 1 hour
    alert_threshold_impact: float = 7.0
    max_concurrent_fetches: int = 10
    enable_web_search: bool = True
    breaking_news_keywords: List[str] = field(default_factory=lambda: [
        "breaking", "just announced", "launches today", "acquires",
        "partnership announced", "shutting down", "major update"
    ])


class StreamingEngine:
    """
    Real-time streaming engine for continuous intelligence gathering.
    
    Runs in the background, collecting and classifying intel continuously.
    Supports callbacks for real-time dashboard updates.
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.event_queue: queue.Queue = queue.Queue()
        self.callbacks: List[Callable[[StreamEvent], None]] = []
        
        # State
        self.is_running = False
        self.last_rss_poll: Optional[datetime] = None
        self.last_web_search: Optional[datetime] = None
        self.seen_urls: set = set()
        self.intel_history: List[ClassifiedIntel] = []
        
        # Threads
        self._poll_thread: Optional[threading.Thread] = None
        self._search_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
    
    def register_callback(self, callback: Callable[[StreamEvent], None]):
        """Register a callback for stream events."""
        self.callbacks.append(callback)
    
    def _emit_event(self, event: StreamEvent):
        """Emit an event to all callbacks."""
        self.event_queue.put(event)
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def start(self):
        """Start the streaming engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start polling thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        
        # Start web search thread (if enabled)
        if self.config.enable_web_search:
            self._search_thread = threading.Thread(target=self._search_loop, daemon=True)
            self._search_thread.start()
        
        # Start event processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self._emit_event(StreamEvent(
            event_type="status",
            timestamp=datetime.now(),
            data={"status": "started", "config": self.config.__dict__},
        ))
        
        print(f"[StreamEngine] Started with {self.config.poll_interval_seconds}s poll interval")
    
    def stop(self):
        """Stop the streaming engine."""
        self.is_running = False
        
        self._emit_event(StreamEvent(
            event_type="status",
            timestamp=datetime.now(),
            data={"status": "stopped"},
        ))
        
        print("[StreamEngine] Stopped")
    
    def _poll_loop(self):
        """Continuous RSS polling loop."""
        while self.is_running:
            try:
                self._do_rss_poll()
            except Exception as e:
                self._emit_event(StreamEvent(
                    event_type="error",
                    timestamp=datetime.now(),
                    data={"error": str(e), "source": "rss_poll"},
                ))
            
            # Sleep until next poll
            time.sleep(self.config.poll_interval_seconds)
    
    def _search_loop(self):
        """Continuous web search loop."""
        while self.is_running:
            try:
                self._do_web_search()
            except Exception as e:
                self._emit_event(StreamEvent(
                    event_type="error",
                    timestamp=datetime.now(),
                    data={"error": str(e), "source": "web_search"},
                ))
            
            # Sleep until next search
            time.sleep(self.config.web_search_interval_seconds)
    
    def _process_loop(self):
        """Process events from the queue."""
        while self.is_running:
            try:
                # Check for high-priority intel
                high_impact = [i for i in self.intel_history[-50:] 
                              if i.impact >= self.config.alert_threshold_impact]
                
                # Emit alerts for new high-impact intel
                for intel in high_impact:
                    if not hasattr(intel, '_alerted'):
                        intel._alerted = True
                        self._emit_event(StreamEvent(
                            event_type="alert",
                            timestamp=datetime.now(),
                            data={
                                "intel": {
                                    "title": intel.title,
                                    "competitor": intel.competitor,
                                    "impact": intel.impact,
                                    "summary": intel.summary,
                                },
                                "reason": f"High impact ({intel.impact}/10)"
                            },
                            priority=2 if intel.impact >= 9 else 1,
                        ))
            except Exception as e:
                pass
            
            time.sleep(10)  # Check every 10 seconds
    
    def _do_rss_poll(self):
        """Perform an RSS poll."""
        self.last_rss_poll = datetime.now()
        
        print(f"[StreamEngine] RSS poll at {self.last_rss_poll.strftime('%H:%M:%S')}")
        
        # Fetch RSS feeds
        articles = fetch_all_feeds_parallel(verbose=False)
        
        # Filter to new articles only
        new_articles = []
        for a in articles:
            if a.url not in self.seen_urls:
                self.seen_urls.add(a.url)
                new_articles.append(a)
        
        if not new_articles:
            print(f"[StreamEngine] No new articles found")
            return
        
        print(f"[StreamEngine] Found {len(new_articles)} new articles")
        
        # Classify new articles
        articles_data = [
            {
                "id": i,
                "title": a.title,
                "url": a.url,
                "raw_snippet": a.raw_snippet,
                "source": getattr(a, 'source_label', getattr(a, 'source', 'unknown')),
                "competitor_id": a.competitor_id,
            }
            for i, a in enumerate(new_articles)
        ]
        
        intel = run_classifier_swarm(articles_data)
        
        if intel:
            self.intel_history.extend(intel)
            
            # Check for breaking news
            for item in intel:
                if self._is_breaking_news(item):
                    self._emit_event(StreamEvent(
                        event_type="alert",
                        timestamp=datetime.now(),
                        data={
                            "intel": {
                                "title": item.title,
                                "competitor": item.competitor,
                                "impact": item.impact,
                                "summary": item.summary,
                            },
                            "reason": "Breaking news detected"
                        },
                        priority=2,
                    ))
            
            # Emit new intel event
            self._emit_event(StreamEvent(
                event_type="new_intel",
                timestamp=datetime.now(),
                data={
                    "count": len(intel),
                    "items": [
                        {
                            "title": i.title,
                            "competitor": i.competitor,
                            "impact": i.impact,
                            "category": i.category,
                            "summary": i.summary,
                        }
                        for i in intel
                    ],
                },
            ))
        
        print(f"[StreamEngine] Classified {len(intel)} intel items")
    
    def _do_web_search(self):
        """Perform a web search sweep."""
        self.last_web_search = datetime.now()
        
        print(f"[StreamEngine] Web search at {self.last_web_search.strftime('%H:%M:%S')}")
        
        # Run search swarm
        swarm = SearchSwarm()
        articles = swarm.search_all(max_results_per_query=3)
        
        # Filter to new articles only
        new_articles = []
        for a in articles:
            if a.url not in self.seen_urls:
                self.seen_urls.add(a.url)
                new_articles.append(a)
        
        if not new_articles:
            print(f"[StreamEngine] No new search results")
            return
        
        print(f"[StreamEngine] Found {len(new_articles)} new search results")
        
        # Classify
        articles_data = [
            {
                "id": i,
                "title": a.title,
                "url": a.url,
                "raw_snippet": a.raw_snippet,
                "source": getattr(a, 'source_label', getattr(a, 'source', 'unknown')),
                "competitor_id": a.competitor_id,
            }
            for i, a in enumerate(new_articles)
        ]
        
        intel = run_classifier_swarm(articles_data)
        
        if intel:
            self.intel_history.extend(intel)
            
            self._emit_event(StreamEvent(
                event_type="new_intel",
                timestamp=datetime.now(),
                data={
                    "count": len(intel),
                    "source": "web_search",
                    "items": [
                        {
                            "title": i.title,
                            "competitor": i.competitor,
                            "impact": i.impact,
                            "category": i.category,
                            "summary": i.summary,
                        }
                        for i in intel
                    ],
                },
            ))
        
        print(f"[StreamEngine] Classified {len(intel)} intel items from search")
    
    def _is_breaking_news(self, intel: ClassifiedIntel) -> bool:
        """Check if intel is breaking news."""
        text = f"{intel.title} {intel.summary}".lower()
        
        for keyword in self.config.breaking_news_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def get_recent_intel(self, limit: int = 50) -> List[ClassifiedIntel]:
        """Get recent intel items."""
        return self.intel_history[-limit:]
    
    def get_stats(self) -> dict:
        """Get streaming engine stats."""
        return {
            "is_running": self.is_running,
            "last_rss_poll": self.last_rss_poll.isoformat() if self.last_rss_poll else None,
            "last_web_search": self.last_web_search.isoformat() if self.last_web_search else None,
            "total_urls_seen": len(self.seen_urls),
            "total_intel": len(self.intel_history),
            "config": {
                "poll_interval": self.config.poll_interval_seconds,
                "search_interval": self.config.web_search_interval_seconds,
            }
        }


# Global streaming engine instance
_engine: Optional[StreamingEngine] = None


def get_streaming_engine() -> StreamingEngine:
    """Get or create the global streaming engine."""
    global _engine
    if _engine is None:
        _engine = StreamingEngine()
    return _engine


def start_streaming(config: Optional[StreamConfig] = None):
    """Start the global streaming engine."""
    global _engine
    _engine = StreamingEngine(config)
    _engine.start()
    return _engine


def stop_streaming():
    """Stop the global streaming engine."""
    global _engine
    if _engine:
        _engine.stop()

