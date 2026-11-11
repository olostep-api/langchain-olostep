"""
LangChain tools for Olostep web scraping API.

Comprehensive tools matching all Olostep API capabilities:
- Scrapes: Extract content from single URLs
- Batches: Scrape multiple URLs in parallel
- Crawls: Autonomously discover and scrape entire websites
- Maps: Extract all URLs from a website
- Searches: Web search results, optionally with each page's content
- Answers: AI-powered web search and question answering
"""

import os
import time
from typing import Literal, List, Dict, Any, Optional, Union
from langchain_core.tools import tool
from langchain_core.exceptions import LangChainException
import requests
import json


class OlostepClient:
    """Client for Olostep API with full feature support."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OLOSTEP_API_KEY")
        if not self.api_key:
            raise ValueError("OLOSTEP_API_KEY environment variable is required")
        self.base_url = "https://api.olostep.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def scrape(
        self, 
        url: str,
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
        parser_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrape a single URL using /v1/scrapes endpoint."""
        payload = {
            "url_to_scrape": url,
            "formats": formats or ["markdown"]
        }
        
        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping
        if parser_id:
            payload["parser"] = {"id": parser_id}
        
        try:
            response = requests.post(
                f"{self.base_url}/scrapes",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep scrape failed: {str(e)}")
    
    def batch_scrape(
        self,
        urls: List[Dict[str, str]],
        formats: Optional[List[str]] = None,
        country: Optional[str] = None,
        wait_before_scraping: int = 0,
        parser_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrape multiple URLs using /v1/batches endpoint."""
        payload = {
            "items": urls
        }
        
        if formats:
            payload["formats"] = formats
        if country:
            payload["country"] = country
        if wait_before_scraping:
            payload["wait_before_scraping"] = wait_before_scraping
        if parser_id:
            payload["parser"] = {"id": parser_id}
        
        try:
            response = requests.post(
                f"{self.base_url}/batches",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep batch scrape failed: {str(e)}")
    
    def answer(
        self,
        task: str,
        json_schema: Optional[Union[Dict, str]] = None
    ) -> Dict[str, Any]:
        """
        Search the web and get AI-powered answers using /v1/answers endpoint.
        
        Args:
            task: Question or task to search for
            json_schema: Optional JSON schema or string describing desired output format
            
        Returns:
            Answer with sources
        """
        payload = {
            "task": task
        }
        
        if json_schema:
            payload["json"] = json_schema
        
        try:
            response = requests.post(
                f"{self.base_url}/answers",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep answer failed: {str(e)}")
    
    def search(
        self,
        query: str,
        limit: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        scrape_results: bool = False
    ) -> Dict[str, Any]:
        """
        Search the web using /v1/searches endpoint.
        
        Args:
            query: Search query in natural language
            limit: Number of results to return (1-25)
            include_domains: Only return results from these domains
            exclude_domains: Never return results from these domains
            scrape_results: Also scrape each result and include its markdown
            
        Returns:
            Search result with a list of links
        """
        payload: Dict[str, Any] = {
            "query": query,
            "limit": limit
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if scrape_results:
            payload["scrape_options"] = {"formats": ["markdown"]}
        
        try:
            response = requests.post(
                f"{self.base_url}/searches",
                headers=self.headers,
                json=payload,
                timeout=90 if scrape_results else 60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep search failed: {str(e)}")
    
    def create_map(
        self,
        url: str,
        search_query: Optional[str] = None,
        top_n: Optional[int] = None,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get all URLs from a website using /v1/maps endpoint.
        
        Args:
            url: Website URL to extract URLs from
            search_query: Optional search query to filter URLs
            top_n: Limit number of URLs returned
            include_urls: Glob patterns to include (e.g., ["/blog/**"])
            exclude_urls: Glob patterns to exclude (e.g., ["/admin/**"])
            
        Returns:
            List of URLs found on the website
        """
        payload = {
            "url": url
        }
        
        if search_query:
            payload["search_query"] = search_query
        if top_n:
            payload["top_n"] = top_n
        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls
        
        try:
            response = requests.post(
                f"{self.base_url}/maps",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep map creation failed: {str(e)}")
    
    def start_crawl(
        self,
        start_url: str,
        max_pages: int = 100,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        include_external: bool = False
    ) -> Dict[str, Any]:
        """
        Start crawling a website using /v1/crawls endpoint.
        
        Args:
            start_url: Starting URL for the crawl
            max_pages: Maximum number of pages to crawl
            include_urls: Glob patterns to include
            exclude_urls: Glob patterns to exclude
            max_depth: Maximum depth to crawl
            include_external: Include external URLs
            
        Returns:
            Crawl job information
        """
        payload = {
            "start_url": start_url,
            "max_pages": max_pages
        }
        
        if include_urls:
            payload["include_urls"] = include_urls
        if exclude_urls:
            payload["exclude_urls"] = exclude_urls
        if max_depth is not None:
            payload["max_depth"] = max_depth
        if include_external:
            payload["include_external"] = include_external
        
        try:
            response = requests.post(
                f"{self.base_url}/crawls",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Olostep crawl failed: {str(e)}")
    
    def get_crawl_status(self, crawl_id: str) -> Dict[str, Any]:
        """Get status of a crawl job."""
        try:
            response = requests.get(
                f"{self.base_url}/crawls/{crawl_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise LangChainException(f"Failed to get crawl status: {str(e)}")
    
    def _get(self, path: str, params: Optional[Any] = None, timeout: int = 60) -> Dict[str, Any]:
        """GET an Olostep API path and return the JSON body."""
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self.headers,
            params=params,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get the status of a batch job."""
        return self._get(f"/batches/{batch_id}", timeout=30)
    
    def get_batch_items(self, batch_id: str, cursor: int = 0, limit: int = 20) -> Dict[str, Any]:
        """List one page of a completed batch's items."""
        return self._get(f"/batches/{batch_id}/items", {"cursor": cursor, "limit": limit})
    
    def get_crawl_pages(self, crawl_id: str, cursor: int = 0, limit: int = 20) -> Dict[str, Any]:
        """List one page of a completed crawl's pages."""
        return self._get(f"/crawls/{crawl_id}/pages", {"cursor": cursor, "limit": limit})
    
    def retrieve(self, retrieve_id: str, formats: List[str]) -> Dict[str, Any]:
        """Get the content of a scraped page by its retrieve_id."""
        return self._get("/retrieve", [("retrieve_id", retrieve_id)] + [("formats[]", f) for f in formats])


def _cap_content(item: Dict[str, Any], content: Any, max_characters: int) -> None:
    """Put content on item, cut to max_characters and marked when it was longer."""
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2)
    content = content or ""
    cap = max(int(max_characters), 1)
    if len(content) > cap:
        item["content"] = content[:cap]
        item["truncated"] = True
        item["full_length"] = len(content)
    else:
        item["content"] = content


def _fetch_contents(
    client: "OlostepClient",
    entries: List[Dict[str, Any]],
    format: str,
    max_characters: int
) -> List[Dict[str, Any]]:
    """Fetch each listed page's content in parallel, capped at max_characters."""
    from concurrent.futures import ThreadPoolExecutor
    
    def fetch(entry: Dict[str, Any]) -> Dict[str, Any]:
        item: Dict[str, Any] = {"url": entry.get("url", "")}
        if entry.get("custom_id"):
            item["custom_id"] = entry["custom_id"]
        retrieve_id = entry.get("retrieve_id")
        if not retrieve_id:
            item["error"] = "No retrieve_id for this page"
            return item
        try:
            data = client.retrieve(retrieve_id, [format])
            _cap_content(item, data.get(f"{format}_content"), max_characters)
            if data.get(f"{format}_hosted_url"):
                item[f"{format}_url"] = data[f"{format}_hosted_url"]
        except Exception as e:
            item["error"] = f"Could not get content: {str(e)}"
        return item
    
    if not entries:
        return []
    with ThreadPoolExecutor(max_workers=min(10, len(entries))) as pool:
        return list(pool.map(fetch, entries))


# LangChain Tools

@tool
def scrape_website(
    url: str,
    format: Literal["markdown", "html", "json", "text"] = "markdown",
    country: Optional[str] = None,
    wait_before_scraping: int = 0,
    parser: Optional[str] = None,
    max_characters: int = 20000,
    api_key: Optional[str] = None
) -> str:
    """Scrape content from any website using Olostep's web scraping API.
    
    Extract clean, LLM-ready content from any website in multiple formats. This tool
    handles JavaScript rendering, anti-bot measures, and provides specialized parsers
    for popular websites like Amazon and Google Search.
    
    Perfect for: Research agents, content analysis, data extraction, and building
    RAG systems with web data.
    
    Args:
        url: Website URL to scrape. Must include protocol (http:// or https://).
            Example: "https://example.com/page"
        format: Output format for the scraped content. Options:
            - "markdown": Clean markdown (default, best for LLMs)
            - "html": Full HTML content
            - "json": Structured JSON data
            - "text": Plain text only
        country: ISO country code for location-specific content. Useful for
            geo-restricted content or localized search results.
            Example: "US", "GB", "CA", "DE"
        wait_before_scraping: Wait time in milliseconds before scraping to allow
            JavaScript to render. Range: 0-10000. Use 2000-5000 for dynamic sites.
            Default: 0
        parser: Optional specialized parser ID for structured extraction from
            popular websites. Available parsers:
            - "@olostep/amazon-product": Amazon product pages
            - "@olostep/google-search": Google search results
        max_characters: Maximum characters of content returned. Longer pages
            are cut and marked with truncated and full_length; the full page is
            at the <format>_url link when the API provides one. Default: 20000
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - content: Scraped content in requested format, up to max_characters
              (with truncated and full_length when it was cut)
            - url: Original URL scraped
            - format: Format used
            - scrape_id: Unique scrape ID for reference
            - metadata: Page metadata (title, description, etc.)
    
    Raises:
        LangChainException: If scraping fails due to network errors, invalid URL,
            or API errors.
    
    Examples:
        Basic scraping with default markdown format:
            >>> from langchain_olostep import scrape_website
            >>> result = scrape_website.invoke({"url": "https://example.com"})
            >>> print(result)
            
        Scraping with JavaScript wait time for dynamic content:
            >>> result = scrape_website.invoke({
            ...     "url": "https://spa-website.com",
            ...     "wait_before_scraping": 3000
            ... })
            
        Using specialized parser for Amazon products:
            >>> result = scrape_website.invoke({
            ...     "url": "https://www.amazon.com/dp/B08N5WRWNW",
            ...     "parser": "@olostep/amazon-product",
            ...     "format": "json"
            ... })
            
        Location-specific scraping:
            >>> result = scrape_website.invoke({
            ...     "url": "https://google.com/search?q=weather",
            ...     "country": "GB"
            ... })
    
    Note:
        For scraping multiple URLs efficiently, use scrape_batch instead.
        See https://docs.olostep.com/api-reference/scrapes for full API documentation.
    """
    try:
        client = OlostepClient(api_key)
        result = client.scrape(
            url=url,
            formats=[format],
            country=country,
            wait_before_scraping=wait_before_scraping,
            parser_id=parser
        )
        
        # Extract content from result
        content_result = result.get("result", {})
        
        # Get the requested format content
        format_key = f"{format}_content"
        content = content_result.get(format_key, "")
        
        # If content is a dict (for json format), convert to string
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        
        # Cap the content so one page can't flood the agent's context
        cap = max(int(max_characters), 1)
        full_length = len(content) if isinstance(content, str) else 0
        if full_length > cap:
            content = content[:cap]
        
        # Build response with metadata
        response = {
            "content": content,
            "url": url,
            "format": format,
            "scrape_id": result.get("retrieve_id", ""),
            "metadata": content_result.get("page_metadata", {})
        }
        if full_length > cap:
            response["truncated"] = True
            response["full_length"] = full_length
        
        # Add hosted URLs if available
        if f"{format}_hosted_url" in content_result:
            response[f"{format}_url"] = content_result[f"{format}_hosted_url"]
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to scrape {url}: {str(e)}")


@tool
def scrape_batch(
    urls: List[str],
    format: Literal["markdown", "html", "json", "text"] = "markdown",
    country: Optional[str] = None,
    wait_before_scraping: int = 0,
    parser: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Scrape multiple websites in parallel using Olostep's batch processing API.
    
    Process up to 100k URLs simultaneously. Batch jobs typically complete in 5-8
    minutes regardless of batch size, making this extremely efficient for large-scale
    data extraction, competitive analysis, and building comprehensive datasets.
    
    Perfect for: Large-scale web scraping, competitor analysis, building training
    datasets, SEO audits, and content aggregation.
    
    Args:
        urls: List of website URLs to scrape. Maximum 100k URLs per batch.
            Each URL must include protocol (http:// or https://).
            Example: ["https://example1.com", "https://example2.com"]
        format: Output format for all URLs. Options:
            - "markdown": Clean markdown (default, best for LLMs)
            - "html": Full HTML content
            - "json": Structured JSON data
            - "text": Plain text only
        country: ISO country code for location-specific content. Applied to all
            URLs in the batch. Example: "US", "GB", "CA"
        wait_before_scraping: Wait time in milliseconds before scraping each URL
            to allow JavaScript to render. Range: 0-10000. Use 2000-5000 for
            dynamic sites. Default: 0
        parser: Optional specialized parser ID applied to all URLs. Use when
            scraping similar page types (e.g., all product pages).
            Example: "@olostep/amazon-product"
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - batch_id: Unique batch ID for tracking
            - status: Current status ("in_progress", "completed", "failed")
            - total_urls: Number of URLs in batch
            - format: Format used
            - urls: List of URLs being processed
    
    Raises:
        LangChainException: If batch creation fails due to network errors, invalid
            URLs, or API errors.
    
    Examples:
        Basic batch scraping:
            >>> from langchain_olostep import scrape_batch
            >>> urls = [
            ...     "https://example1.com",
            ...     "https://example2.com",
            ...     "https://example3.com"
            ... ]
            >>> result = scrape_batch.invoke({"urls": urls})
            >>> import json
            >>> batch_info = json.loads(result)
            >>> print(f"Batch ID: {batch_info['batch_id']}")
            
        Batch with custom format and country:
            >>> result = scrape_batch.invoke({
            ...     "urls": ["https://shop1.com", "https://shop2.com"],
            ...     "format": "json",
            ...     "country": "US",
            ...     "wait_before_scraping": 2000
            ... })
            
        Using parser for structured data extraction:
            >>> product_urls = [
            ...     "https://amazon.com/dp/PRODUCT1",
            ...     "https://amazon.com/dp/PRODUCT2",
            ... ]
            >>> result = scrape_batch.invoke({
            ...     "urls": product_urls,
            ...     "parser": "@olostep/amazon-product"
            ... })
    
    Note:
        Batch processing is asynchronous. The tool returns immediately with a
        batch_id. Pass it to get_batch_results to get each page's content.
        See https://docs.olostep.com/api-reference/batches for full documentation.
    """
    try:
        client = OlostepClient(api_key)
        
        # Convert URLs to batch format
        batch_items = [
            {"url": url, "custom_id": f"url_{i}"}
            for i, url in enumerate(urls)
        ]
        
        result = client.batch_scrape(
            urls=batch_items,
            formats=[format],
            country=country,
            wait_before_scraping=wait_before_scraping,
            parser_id=parser
        )
        
        response = {
            "batch_id": result.get("batch_id", result.get("id", "")),
            "status": result.get("status", "in_progress"),
            "total_urls": len(urls),
            "format": format,
            "urls": urls
        }
        
        if country:
            response["country"] = country
        if parser:
            response["parser"] = parser
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to scrape batch: {str(e)}")


@tool
def answer_question(
    task: str,
    json_schema: Optional[Union[Dict, str]] = None,
    api_key: Optional[str] = None
) -> str:
    """Search the web and get AI-powered answers with structured output and sources.
    
    Ground your AI agents on real-world, up-to-date data. This tool searches the web,
    analyzes multiple sources, and returns structured answers in your desired format.
    It's perfect for research, data enrichment, fact-checking, and building AI agents
    that need current web information.
    
    Perfect for: Research agents, data enrichment pipelines, market intelligence,
    competitive analysis, and fact-checking.
    
    Args:
        task: Question or research task to answer. Be specific and clear.
            Examples:
            - "What is the latest product from Apple?"
            - "Find the CEO and founding year of Stripe"
            - "What are the top 3 competitors of Shopify?"
        json_schema: Optional JSON schema defining the desired output structure.
            Can be a dictionary or string. The AI will fill in the fields based on
            web research. Use empty strings as placeholders.
            Examples:
            - {"company": "", "ceo": "", "founded": ""}
            - {"book_title": "", "author": "", "release_date": "", "rating": ""}
            If not provided, returns unstructured answer.
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - answer: Structured answer matching your schema (or unstructured text)
            - task: Original task/question
            - sources: List of source URLs used to generate the answer
            - answer_id: Unique answer ID for reference
    
    Raises:
        LangChainException: If answer generation fails due to network errors or
            API errors.
    
    Examples:
        Simple question without schema:
            >>> from langchain_olostep import answer_question
            >>> result = answer_question.invoke({
            ...     "task": "What is the capital of France?"
            ... })
            >>> import json
            >>> answer = json.loads(result)
            >>> print(answer["answer"])
            
        Structured data extraction with schema:
            >>> result = answer_question.invoke({
            ...     "task": "Find information about Stripe",
            ...     "json_schema": {
            ...         "company": "",
            ...         "ceo": "",
            ...         "headquarters": "",
            ...         "founded_year": ""
            ...     }
            ... })
            >>> answer = json.loads(result)
            >>> print(f"CEO: {answer['answer']['ceo']}")
            >>> print(f"Sources: {answer['sources']}")
            
        Data enrichment for company research:
            >>> result = answer_question.invoke({
            ...     "task": "What is the latest funding round for Anthropic?",
            ...     "json_schema": {
            ...         "amount": "",
            ...         "date": "",
            ...         "investors": "",
            ...         "valuation": ""
            ...     }
            ... })
            
        Product research with sources:
            >>> result = answer_question.invoke({
            ...     "task": "What are the specs of iPhone 15 Pro?",
            ...     "json_schema": {
            ...         "model": "",
            ...         "release_date": "",
            ...         "price": "",
            ...         "key_features": []
            ...     }
            ... })
    
    Note:
        The AI returns "NOT_FOUND" or similar indicators when information cannot be
        confidently verified. Always check sources for reliability.
        See https://docs.olostep.com/api-reference/answers for full documentation.
    """
    try:
        client = OlostepClient(api_key)
        result = client.answer(
            task=task,
            json_schema=json_schema
        )
        
        # Extract JSON content from result
        result_data = result.get("result", {})
        json_content = result_data.get("json_content", "")
        
        # Parse JSON content if it's a string
        if isinstance(json_content, str) and json_content:
            try:
                json_content = json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        response = {
            "answer": json_content,
            "task": task,
            "sources": result_data.get("sources", []),
            "answer_id": result.get("id", "")
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to get answer for task '{task}': {str(e)}")


@tool
def search_web(
    query: str,
    limit: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    scrape_results: bool = False,
    max_characters: int = 20000,
    api_key: Optional[str] = None
) -> str:
    """Search the web with Olostep and return the matching pages.
    
    Use this when you need web pages on a topic and do not have a URL yet. Returns
    each result's URL, title and description. With scrape_results=True, each result
    also includes the page's content as markdown, so the agent can read the pages
    without a separate scrape. For a direct answer with sources, use answer_question.
    
    Args:
        query: What to search for, in natural language.
            Example: "best open source vector databases 2026"
        limit: Number of results to return, from 1 to 25. Default: 5
        include_domains: Only return results from these domains.
            Example: ["docs.python.org", "github.com"]
        exclude_domains: Never return results from these domains.
            Example: ["pinterest.com"]
        scrape_results: Also scrape each result and include its content as
            markdown. Each scraped page uses extra credits. Default: False
        max_characters: Maximum characters of markdown kept per page when
            scrape_results is True. Longer pages are cut and marked with
            truncated and full_length; use scrape_website for the full page.
            Default: 20000
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - query: The search query
            - search_id: Unique search ID for reference
            - results: List of results, each with url, title and description,
              plus markdown_content when scrape_results is True (and
              truncated/full_length when the page was cut)
            - total_results: Number of results returned
    
    Raises:
        LangChainException: If the search fails due to network or API errors.
    
    Examples:
        Basic search:
            >>> from langchain_olostep import search_web
            >>> result = search_web.invoke({"query": "LangChain agents tutorial"})
            
        Search one site and read the pages in the same call:
            >>> result = search_web.invoke({
            ...     "query": "rate limits",
            ...     "include_domains": ["docs.olostep.com"],
            ...     "scrape_results": True
            ... })
    
    Note:
        See https://docs.olostep.com/features/search for full documentation.
    """
    try:
        client = OlostepClient(api_key)
        result = client.search(
            query=query,
            limit=min(max(int(limit), 1), 25),
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            scrape_results=scrape_results
        )
        
        links = (result.get("result") or {}).get("links") or []
        cap = max(int(max_characters), 1)
        results = []
        for link in links:
            item = {
                "url": link.get("url", ""),
                "title": link.get("title", ""),
                "description": link.get("description", "")
            }
            content = link.get("markdown_content")
            if content:
                if len(content) > cap:
                    item["markdown_content"] = content[:cap]
                    item["truncated"] = True
                    item["full_length"] = len(content)
                else:
                    item["markdown_content"] = content
            results.append(item)
        
        response = {
            "query": query,
            "search_id": result.get("id", ""),
            "results": results,
            "total_results": len(results)
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to search the web for '{query}': {str(e)}")


@tool
def extract_urls(
    url: str,
    search_query: Optional[str] = None,
    top_n: Optional[int] = None,
    include_urls: Optional[List[str]] = None,
    exclude_urls: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    """Extract all URLs from a website for sitemap generation and content discovery.
    
    Discover and map the complete structure of any website. This tool finds all pages,
    which is perfect for preparing URLs for batch scraping, SEO audits, competitive
    analysis, or understanding website architecture. Can discover up to ~100,000 URLs
    in a single call.
    
    Perfect for: SEO audits, sitemap generation, competitive analysis, preparing batch
    scraping jobs, and website structure analysis.
    
    Args:
        url: Base website URL to extract URLs from. Must include protocol
            (http:// or https://). The tool will discover all linked pages.
            Example: "https://example.com"
        search_query: Optional search query to filter URLs by relevance. Uses
            semantic search to find URLs matching the query.
            Example: "blog posts about AI"
        top_n: Maximum number of URLs to return. Useful for limiting results on
            large websites. If not specified, returns all discovered URLs (up to
            ~100,000).
        include_urls: List of glob patterns to include. Only URLs matching these
            patterns will be returned. Use ** for wildcard matching.
            Examples:
            - ["/blog/**"] - Only blog posts
            - ["/product/**", "/category/**"] - Products and categories
            - ["/**"] - All pages (default behavior)
        exclude_urls: List of glob patterns to exclude. URLs matching these
            patterns will be filtered out. Use ** for wildcard matching.
            Examples:
            - ["/admin/**", "/private/**"] - Exclude admin and private pages
            - ["/tag/**"] - Exclude tag pages
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - map_id: Unique map ID for reference
            - url: Base URL that was mapped
            - total_urls: Number of URLs discovered
            - urls: List of discovered URLs
    
    Raises:
        LangChainException: If URL extraction fails due to network errors, invalid
            URL, or API errors.
    
    Examples:
        Extract all URLs from a website:
            >>> from langchain_olostep import extract_urls
            >>> result = extract_urls.invoke({"url": "https://example.com"})
            >>> import json
            >>> data = json.loads(result)
            >>> print(f"Found {data['total_urls']} URLs")
            >>> for url in data['urls'][:5]:
            ...     print(url)
            
        Extract only blog URLs:
            >>> result = extract_urls.invoke({
            ...     "url": "https://example.com",
            ...     "include_urls": ["/blog/**"]
            ... })
            
        Extract product pages, exclude admin:
            >>> result = extract_urls.invoke({
            ...     "url": "https://store.com",
            ...     "include_urls": ["/product/**"],
            ...     "exclude_urls": ["/admin/**", "/checkout/**"],
            ...     "top_n": 100
            ... })
            
        Semantic search for specific content:
            >>> result = extract_urls.invoke({
            ...     "url": "https://docs.example.com",
            ...     "search_query": "API authentication",
            ...     "top_n": 10
            ... })
            
        Prepare URLs for batch scraping:
            >>> # First, discover all product URLs
            >>> urls_result = extract_urls.invoke({
            ...     "url": "https://ecommerce.com",
            ...     "include_urls": ["/product/**"]
            ... })
            >>> # Then scrape them all with scrape_batch
            >>> from langchain_olostep import scrape_batch
            >>> urls_data = json.loads(urls_result)
            >>> batch_result = scrape_batch.invoke({
            ...     "urls": urls_data["urls"]
            ... })
    
    Note:
        This tool is extremely fast, typically completing in seconds even for large
        websites. Combine with scrape_batch for efficient large-scale extraction.
        See https://docs.olostep.com/api-reference/maps for full documentation.
    """
    try:
        client = OlostepClient(api_key)
        result = client.create_map(
            url=url,
            search_query=search_query,
            top_n=top_n,
            include_urls=include_urls,
            exclude_urls=exclude_urls
        )
        
        urls = result.get("urls", [])
        
        response = {
            "map_id": result.get("id", result.get("map_id", "")),
            "url": url,
            "total_urls": len(urls),
            "urls": urls
        }
        
        if search_query:
            response["search_query"] = search_query
        if top_n:
            response["top_n"] = top_n
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to extract URLs from {url}: {str(e)}")


@tool
def crawl_website(
    start_url: str,
    max_pages: int = 100,
    include_urls: Optional[List[str]] = None,
    exclude_urls: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    include_external: bool = False,
    api_key: Optional[str] = None
) -> str:
    """Autonomously crawl and scrape entire websites by following links.
    
    Perfect for comprehensive website extraction without knowing the structure
    beforehand. The crawler intelligently follows links, discovers pages, and scrapes
    content automatically. Ideal for documentation sites, blogs, knowledge bases, and
    building complete website archives.
    
    Perfect for: Documentation scraping, building knowledge bases, comprehensive
    content extraction, blog archiving, and creating training datasets.
    
    Args:
        start_url: Starting URL for the crawl. The crawler will begin here and
            follow links to discover other pages. Must include protocol
            (http:// or https://). Example: "https://docs.example.com"
        max_pages: Maximum number of pages to crawl. Controls crawl scope and
            cost. Default: 100. Increase for larger sites (up to thousands).
        include_urls: List of glob patterns to include. Only URLs matching these
            patterns will be crawled. Use ** for wildcard matching.
            Examples:
            - ["/**"] - Crawl all pages (default)
            - ["/docs/**"] - Only documentation pages
            - ["/blog/**", "/news/**"] - Only blog and news pages
        exclude_urls: List of glob patterns to exclude. URLs matching these
            patterns will be skipped during crawling. Use ** for wildcard matching.
            Examples:
            - ["/admin/**", "/private/**"] - Skip admin and private pages
            - ["/tag/**", "/category/**"] - Skip tag and category pages
        max_depth: Maximum link depth to crawl from start_url. Limits how far
            the crawler goes from the starting point.
            Example: max_depth=2 means start_url → page1 → page2 (stops here)
            If not specified, crawls until max_pages is reached.
        include_external: Whether to follow and crawl external links (links to
            other domains). Default: False (only crawls the same domain).
            Set to True to crawl referenced external sites.
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string containing:
            - crawl_id: Unique crawl ID for tracking
            - status: Current status ("in_progress", "completed", "failed")
            - start_url: Starting URL of the crawl
            - max_pages: Maximum pages to crawl
            - pages_crawled: Number of pages crawled so far
    
    Raises:
        LangChainException: If crawl initiation fails due to network errors, invalid
            URL, or API errors.
    
    Examples:
        Crawl an entire documentation site:
            >>> from langchain_olostep import crawl_website
            >>> result = crawl_website.invoke({
            ...     "start_url": "https://docs.example.com"
            ... })
            >>> import json
            >>> crawl_info = json.loads(result)
            >>> print(f"Crawl ID: {crawl_info['crawl_id']}")
            
        Crawl with URL filters:
            >>> result = crawl_website.invoke({
            ...     "start_url": "https://example.com",
            ...     "max_pages": 200,
            ...     "include_urls": ["/docs/**"],
            ...     "exclude_urls": ["/admin/**", "/api/**"]
            ... })
            
        Limited depth crawl for blog:
            >>> result = crawl_website.invoke({
            ...     "start_url": "https://blog.example.com",
            ...     "max_pages": 50,
            ...     "max_depth": 3
            ... })
            
        Crawl with external links:
            >>> result = crawl_website.invoke({
            ...     "start_url": "https://resource-hub.com",
            ...     "max_pages": 100,
            ...     "include_external": True
            ... })
            
        Build a comprehensive knowledge base:
            >>> # Crawl entire documentation site
            >>> result = crawl_website.invoke({
            ...     "start_url": "https://docs.example.com",
            ...     "max_pages": 500,
            ...     "exclude_urls": ["/v1/**", "/v2/**"]  # Exclude old versions
            ... })
            >>> # The crawled content can then be processed for RAG
    
    Note:
        Crawling is asynchronous. The tool returns immediately with a crawl_id.
        Pass it to get_crawl_results to get each page's content. Large crawls
        may take several minutes to complete.
        See https://docs.olostep.com/api-reference/crawls for full documentation.
    """
    try:
        client = OlostepClient(api_key)
        result = client.start_crawl(
            start_url=start_url,
            max_pages=max_pages,
            include_urls=include_urls,
            exclude_urls=exclude_urls,
            max_depth=max_depth,
            include_external=include_external
        )
        
        response = {
            "crawl_id": result.get("id", ""),
            "status": result.get("status", "in_progress"),
            "start_url": start_url,
            "max_pages": max_pages,
            "pages_crawled": result.get("pages_crawled", 0)
        }
        
        if max_depth is not None:
            response["max_depth"] = max_depth
        if include_external:
            response["include_external"] = include_external
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        raise LangChainException(f"Failed to start crawl for {start_url}: {str(e)}")


def _job_results(
    kind: str,
    job_id: str,
    status: Dict[str, Any],
    listing: Dict[str, Any],
    entries: List[Dict[str, Any]],
    cursor: int,
    total: Optional[int],
    client: "OlostepClient",
    format: str,
    max_characters: int
) -> str:
    """Build the JSON result for get_batch_results and get_crawl_results."""
    next_cursor = listing.get("cursor")
    # The API returns a cursor whenever a page is full, even if nothing is left.
    first_page_has_all = not cursor and isinstance(total, int) and len(entries) >= total
    has_more = bool(entries) and next_cursor is not None and not first_page_has_all
    response: Dict[str, Any] = {
        f"{kind}_id": status.get("id", job_id),
        "status": "completed",
        "items_returned": len(entries),
        "has_more": has_more,
        "items": _fetch_contents(client, entries, format, max_characters),
    }
    if has_more:
        response["next_cursor"] = next_cursor
    return json.dumps(response, indent=2)


def _still_running(kind: str, job_id: str, status: Dict[str, Any]) -> str:
    """Build the JSON result for a batch or crawl that has not finished."""
    state = status.get("status", "unknown")
    return json.dumps({
        f"{kind}_id": status.get("id", job_id),
        "status": state,
        "message": f"The {kind} is {state}, not completed yet. Call get_{kind}_results again in a minute.",
    }, indent=2)


@tool
def get_batch_results(
    batch_id: str,
    format: Literal["markdown", "html", "json", "text"] = "markdown",
    items_limit: int = 20,
    cursor: int = 0,
    max_characters: int = 20000,
    api_key: Optional[str] = None
) -> str:
    """Get the scraped content of a batch started with scrape_batch.
    
    Returns each page's content, a set of pages per call. If the batch is still
    running, returns its status so you can call again later. When has_more is
    true, call again with next_cursor to get the next pages.
    
    Args:
        batch_id: The batch_id returned by scrape_batch.
        format: Content format to return: "markdown" (default), "html",
            "json" or "text".
        items_limit: Number of pages to return per call, from 1 to 100.
            Default: 20
        cursor: Pass next_cursor from the previous call to get the next pages.
            Default: 0 (first page)
        max_characters: Maximum characters of content per page. Longer pages
            are cut and marked with truncated and full_length. Default: 20000
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string with batch_id, status, items_returned, has_more,
        next_cursor (when has_more) and items, each with url, custom_id and
        content.
    
    Examples:
        >>> from langchain_olostep import scrape_batch, get_batch_results
        >>> import json
        >>> batch = json.loads(scrape_batch.invoke({"urls": ["https://example.com"]}))
        >>> result = get_batch_results.invoke({"batch_id": batch["batch_id"]})
    """
    try:
        client = OlostepClient(api_key)
        status = client.get_batch(batch_id)
        if str(status.get("status", "")).lower() != "completed":
            return _still_running("batch", batch_id, status)
        limit = min(max(int(items_limit), 1), 100)
        listing = client.get_batch_items(batch_id, cursor=cursor, limit=limit)
        entries = listing.get("items") or []
        total = status.get("total_urls")
        return _job_results("batch", batch_id, status, listing, entries, cursor, total,
                            client, format, max_characters)
    except Exception as e:
        raise LangChainException(f"Failed to get results for batch {batch_id}: {str(e)}")


@tool
def get_crawl_results(
    crawl_id: str,
    format: Literal["markdown", "html", "json", "text"] = "markdown",
    items_limit: int = 20,
    cursor: int = 0,
    max_characters: int = 20000,
    api_key: Optional[str] = None
) -> str:
    """Get the content of the pages found by a crawl started with crawl_website.
    
    Returns each page's content, a set of pages per call. If the crawl is still
    running, returns its status so you can call again later. When has_more is
    true, call again with next_cursor to get the next pages.
    
    Args:
        crawl_id: The crawl_id returned by crawl_website.
        format: Content format to return: "markdown" (default), "html",
            "json" or "text".
        items_limit: Number of pages to return per call, from 1 to 100.
            Default: 20
        cursor: Pass next_cursor from the previous call to get the next pages.
            Default: 0 (first page)
        max_characters: Maximum characters of content per page. Longer pages
            are cut and marked with truncated and full_length. Default: 20000
        api_key: Olostep API key. If not provided, uses OLOSTEP_API_KEY
            environment variable.
    
    Returns:
        str: JSON string with crawl_id, status, items_returned, has_more,
        next_cursor (when has_more) and items, each with url and content.
    
    Examples:
        >>> from langchain_olostep import crawl_website, get_crawl_results
        >>> import json
        >>> crawl = json.loads(crawl_website.invoke({"start_url": "https://docs.example.com", "max_pages": 10}))
        >>> result = get_crawl_results.invoke({"crawl_id": crawl["crawl_id"]})
    """
    try:
        client = OlostepClient(api_key)
        status = client.get_crawl_status(crawl_id)
        if str(status.get("status", "")).lower() != "completed":
            return _still_running("crawl", crawl_id, status)
        limit = min(max(int(items_limit), 1), 100)
        listing = client.get_crawl_pages(crawl_id, cursor=cursor, limit=limit)
        entries = listing.get("pages") or listing.get("items") or []
        total = status.get("pages_count")
        return _job_results("crawl", crawl_id, status, listing, entries, cursor, total,
                            client, format, max_characters)
    except Exception as e:
        raise LangChainException(f"Failed to get results for crawl {crawl_id}: {str(e)}")
