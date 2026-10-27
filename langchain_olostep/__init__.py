"""
LangChain Olostep Integration

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data from any website.

Features:
- Scrapes: Extract content from any website with JavaScript rendering support
- Batches: Process up to 10,000 URLs in parallel (5-8 minutes)
- Crawls: Autonomously discover and scrape entire websites
- Maps: Extract all URLs from a website for site structure analysis
- Answers: AI-powered web search with natural language queries
- Multiple Formats: Support for Markdown, HTML, JSON, and plain text
"""

from .tools import (
    scrape_website,
    scrape_batch,
    answer_question,
    extract_urls,
    crawl_website,
)

__version__ = "0.3.5"
__all__ = [
    "scrape_website",
    "scrape_batch",
    "answer_question",
    "extract_urls",
    "crawl_website",
]
