"""
Tools module - Functions exposed to agents for data access and I/O operations.
"""

from radar.tools.rss import fetch_rss
from radar.tools.html import fetch_html_excerpt
from radar.tools.db_tools import (
    store_articles,
    get_unprocessed_articles,
    store_intel,
    get_recent_intel_for_dedup,
    store_novelty_scores,
    get_intel_for_domain,
    store_annotations,
    create_report_file,
)
from radar.tools.vector import embed_and_index_intel, search_similar_intel

__all__ = [
    "fetch_rss",
    "fetch_html_excerpt",
    "store_articles",
    "get_unprocessed_articles",
    "store_intel",
    "get_recent_intel_for_dedup",
    "store_novelty_scores",
    "get_intel_for_domain",
    "store_annotations",
    "create_report_file",
    "embed_and_index_intel",
    "search_similar_intel",
]

