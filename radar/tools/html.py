"""
HTML fetching and text extraction tool.

Fetches web pages and extracts clean text content.
"""
from __future__ import annotations

import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 10.0

# User agent for requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Tags to remove from HTML
REMOVE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", 
    "form", "iframe", "noscript", "svg", "img", "video", "audio"
]

# Content-containing tags to prioritize
CONTENT_TAGS = ["article", "main", "div.content", "div.post", "div.article"]


def clean_text(text: str) -> str:
    """Clean extracted text by normalizing whitespace."""
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)
    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_text_from_html(html: str, max_chars: int = 4000) -> str:
    """
    Extract clean text content from HTML.
    
    Args:
        html: Raw HTML content
        max_chars: Maximum characters to return
    
    Returns:
        Clean text content
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Remove unwanted tags
    for tag in REMOVE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()
    
    # Try to find main content area
    content = None
    for selector in CONTENT_TAGS:
        if "." in selector:
            tag, class_name = selector.split(".")
            content = soup.find(tag, class_=class_name)
        else:
            content = soup.find(selector)
        if content:
            break
    
    # Fall back to body or entire document
    if not content:
        content = soup.find("body") or soup
    
    # Extract and clean text
    text = content.get_text(separator="\n", strip=True)
    text = clean_text(text)
    
    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text


@tool
def fetch_html_excerpt(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a web page and extract clean text content.
    
    Args:
        url: The URL to fetch
        max_chars: Maximum characters to return (default 4000)
    
    Returns:
        Clean text excerpt from the page, or error message
    """
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return f"[Non-HTML content: {content_type}]"
            
            html = response.text
            return extract_text_from_html(html, max_chars)
    
    except httpx.TimeoutException:
        return f"[Error: Request timed out for {url}]"
    except httpx.HTTPStatusError as e:
        return f"[Error: HTTP {e.response.status_code} for {url}]"
    except Exception as e:
        return f"[Error fetching {url}: {str(e)}]"


def fetch_html_excerpt_sync(url: str, max_chars: int = 4000) -> str:
    """
    Synchronous version of fetch_html_excerpt for direct use.
    
    Args:
        url: The URL to fetch
        max_chars: Maximum characters to return
    
    Returns:
        Clean text excerpt from the page
    """
    return fetch_html_excerpt.invoke({"url": url, "max_chars": max_chars})

