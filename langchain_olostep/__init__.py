"""
LangChain Olostep Integration

The most reliable and cost-effective web search, scraping and crawling API for AI.
Build intelligent agents that can search, scrape, analyze, and structure data from any website.

Features:
- Scrapes: Extract content from any website with JavaScript rendering support
- Batches: Process up to 100k URLs in parallel (5-8 minutes)
- Crawls: Autonomously discover and scrape entire websites
- Maps: Extract all URLs from a website for site structure analysis
- Searches: Web search results, optionally with each page's content
- Answers: AI-powered web search with natural language queries
- Multiple Formats: Support for Markdown, HTML, JSON, and plain text
"""

from .tools import (
    scrape_website,
    scrape_batch,
    answer_question,
    search_web,
    extract_urls,
    crawl_website,
    get_batch_results,
    get_crawl_results,
)

__version__ = "0.5.0"
__all__ = [
    "scrape_website",
    "scrape_batch",
    "answer_question",
    "search_web",
    "extract_urls",
    "crawl_website",
    "get_batch_results",
    "get_crawl_results",
]
