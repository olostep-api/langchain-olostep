# LangChain Olostep Integration

**The most reliable and cost-effective web search, scraping and crawling API for AI.**

Build intelligent agents that can search, scrape, analyze, and structure data from any website. Perfect for LangChain and LangGraph applications.

## Features

This integration provides access to all 5 Olostep API endpoints:

### 1. Scrapes (`/v1/scrapes`)
Extract content from any single URL in multiple formats (Markdown, HTML, JSON, text). Handles JavaScript rendering, anti-scraping measures, and supports specialized parsers for specific websites.

### 2. Batches (`/v1/batches`)
Process up to 10,000 URLs in parallel. Batch jobs typically complete in 5-8 minutes regardless of batch size. Perfect for large-scale data extraction and competitor analysis.

### 3. Answers (`/v1/answers`)
AI-powered web search with natural language queries. Ask questions and get structured answers with sources. Ground your AI products on real-world data and facts. Perfect for data enrichment and research.

### 4. Maps (`/v1/maps`)
Extract all URLs from a website for site structure analysis and content discovery. Can discover up to ~100,000 URLs in a single call. Perfect for SEO audits and preparing URLs for batch processing.

### 5. Crawls (`/v1/crawls`)
Autonomously discover and scrape entire websites by following links. Perfect for documentation sites, blogs, and comprehensive website data extraction.

### Core Capabilities
- **Multiple Formats**: Markdown, HTML, JSON, and plain text
- **JavaScript Rendering**: Automatic handling of dynamic content
- **Specialized Parsers**: Custom parsers for Amazon, Google Search, and more
- **Location-Specific**: Scrape with country-specific settings
- **LangGraph Ready**: Perfect for building complex AI agent workflows
- **Cost-Effective**: Pay only for what you use

## Installation

```bash
pip install langchain-olostep
```

## Setup

Set your Olostep API key:

```bash
export OLOSTEP_API_KEY="your_olostep_api_key_here"
```

Get your API key from https://olostep.com/dashboard

## Quick Start

### 1. Basic Web Scraping

```python
from langchain_olostep import scrape_website
import asyncio

# Scrape a website
content = asyncio.run(scrape_website.ainvoke({
    "url": "https://example.com",
    "format": "markdown"
}))

print(content)
```

### 2. Batch Scraping

```python
from langchain_olostep import scrape_batch
import asyncio

# Scrape multiple URLs in parallel
result = asyncio.run(scrape_batch.ainvoke({
    "urls": [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com"
    ],
    "format": "markdown"
}))

print(result)
```

### 3. AI-Powered Web Search & Q&A

```python
from langchain_olostep import answer_question
import asyncio

# Ask a question and get structured answer
result = asyncio.run(answer_question.ainvoke({
    "task": "What is the latest book by J.K. Rowling?",
    "json_schema": {
        "book_title": "",
        "author": "",
        "release_date": ""
    }
}))

print(result)
# Returns: {"book_title": "The Hallmarked Man", "author": "J.K. Rowling", ...}
# Also includes sources used
```

### 4. Extract URLs from Website

```python
from langchain_olostep import extract_urls
import asyncio

# Get all URLs from a website
result = asyncio.run(extract_urls.ainvoke({
    "url": "https://example.com",
    "include_urls": ["/blog/**"],  # Only blog posts
    "top_n": 100
}))

print(result)
```

### 5. Crawl Entire Website

```python
from langchain_olostep import crawl_website
import asyncio

# Autonomously crawl and scrape a website
result = asyncio.run(crawl_website.ainvoke({
    "start_url": "https://docs.example.com",
    "max_pages": 100,
    "exclude_urls": ["/admin/**"]
}))

print(result)
```

## Usage with LangChain Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_olostep import (
    scrape_website,
    answer_question,
    extract_urls
)

# Create agent with Olostep tools
tools = [scrape_website, answer_question, extract_urls]
llm = ChatOpenAI(model="gpt-4o-mini")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.run("""
Research the company at https://company.com:
1. Scrape their about page
2. Search for their latest funding round
3. Extract all their product pages
""")

print(result)
```

## Usage with LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain_olostep import (
    scrape_website,
    answer_question,
    extract_urls
)
from langchain_openai import ChatOpenAI

# Build a research agent workflow
def create_research_agent():
    workflow = StateGraph(dict)
    
    def discover_pages(state):
        # Extract all URLs from target site
        result = extract_urls.invoke({
            "url": state["target_url"],
            "include_urls": ["/product/**"],
            "top_n": 50
        })
        state["urls"] = json.loads(result)["urls"]
        return state
    
    def scrape_pages(state):
        # Scrape discovered pages in batch
        result = scrape_batch.invoke({
            "urls": state["urls"],
            "format": "markdown"
        })
        state["batch_id"] = json.loads(result)["batch_id"]
        return state
    
    def answer_questions(state):
        # Use AI to answer questions about the data
        result = answer_question.invoke({
            "task": state["research_question"],
            "json_schema": state["desired_format"]
        })
        state["answer"] = json.loads(result)["answer"]
        return state
    
    workflow.add_node("discover", discover_pages)
    workflow.add_node("scrape", scrape_pages)
    workflow.add_node("analyze", answer_questions)
    
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "scrape")
    workflow.add_edge("scrape", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()

# Use the agent
agent = create_research_agent()
result = agent.invoke({
    "target_url": "https://store.com",
    "research_question": "What are the top 5 most expensive products?",
    "desired_format": {"products": [{"name": "", "price": "", "url": ""}]}
})
```

## Available Tools

### `scrape_website`
Scrape content from a single URL. Supports markdown, HTML, JSON, and text formats.

**Parameters:**
- `url` (required): Website URL to scrape
- `format`: Output format (markdown, html, json, text). Default: "markdown"
- `country`: Country code for location-specific content (e.g., "US", "GB", "CA")
- `wait_before_scraping`: Wait time in milliseconds for JavaScript rendering
- `parser`: Optional parser ID for specialized extraction (e.g., "@olostep/amazon-product")

### `scrape_batch`
Scrape multiple URLs in parallel (up to 10,000 at once).

**Parameters:**
- `urls` (required): List of URLs to scrape
- `format`: Output format for all URLs. Default: "markdown"
- `country`: Country code for location-specific content
- `wait_before_scraping`: Wait time in milliseconds for JavaScript rendering
- `parser`: Optional parser ID for specialized extraction

### `answer_question`
Search the web and get AI-powered answers with sources.

**Parameters:**
- `task` (required): Question or task to search for
- `json_schema`: Optional JSON schema dict/string describing desired output format

**Examples:**
```python
# Simple question
answer_question.invoke({"task": "What is the capital of France?"})

# With structured output
answer_question.invoke({
    "task": "Find the CEO of Stripe",
    "json_schema": {"ceo_name": "", "founded_year": ""}
})
```

### `extract_urls`
Extract all URLs from a website for site structure analysis.

**Parameters:**
- `url` (required): Website URL to extract URLs from
- `search_query`: Optional search query to filter URLs
- `top_n`: Limit number of URLs returned
- `include_urls`: Glob patterns to include (e.g., ["/blog/**"])
- `exclude_urls`: Glob patterns to exclude (e.g., ["/admin/**"])

### `crawl_website`
Autonomously discover and scrape entire websites.

**Parameters:**
- `start_url` (required): Starting URL for the crawl
- `max_pages`: Maximum number of pages to crawl. Default: 100
- `include_urls`: Glob patterns to include
- `exclude_urls`: Glob patterns to exclude
- `max_depth`: Maximum depth to crawl from start_url
- `include_external`: Include external URLs. Default: False

## Advanced Examples

### Data Enrichment

```python
from langchain_olostep import answer_question

# Enrich company data from spreadsheet
companies = ["Stripe", "Shopify", "Square"]

for company in companies:
    result = answer_question.invoke({
        "task": f"Find information about {company}",
        "json_schema": {
            "ceo": "",
            "headquarters": "",
            "employee_count": "",
            "latest_funding": ""
        }
    })
    print(f"{company}: {result}")
```

### E-commerce Product Scraping

```python
from langchain_olostep import scrape_website

# Scrape Amazon product with specialized parser
result = scrape_website.invoke({
    "url": "https://www.amazon.com/dp/PRODUCT_ID",
    "parser": "@olostep/amazon-product",
    "format": "json"
})
# Returns structured product data: price, title, rating, etc.
```

### SEO Audit

```python
from langchain_olostep import extract_urls, scrape_batch

# 1. Discover all pages
urls_result = extract_urls.invoke({
    "url": "https://yoursite.com",
    "top_n": 1000
})

# 2. Scrape all pages to analyze
urls = json.loads(urls_result)["urls"]
batch_result = scrape_batch.invoke({
    "urls": urls,
    "format": "html"
})
```

### Documentation Scraping

```python
from langchain_olostep import crawl_website

# Crawl entire docs site
result = crawl_website.invoke({
    "start_url": "https://docs.example.com",
    "max_pages": 500,
    "include_urls": ["/docs/**"],
    "exclude_urls": ["/api/**", "/v1/**"]  # Exclude old versions
})
```

## Error Handling

```python
from langchain_core.exceptions import LangChainException

try:
    result = await scrape_website.ainvoke({
        "url": "https://example.com"
    })
except LangChainException as e:
    print(f"Scraping failed: {e}")
```

## Environment Variables

- `OLOSTEP_API_KEY`: Your Olostep API key (required)

## Specialized Parsers

Olostep provides pre-built parsers for popular websites:

- `@olostep/amazon-product`: Amazon product pages
- `@olostep/google-search`: Google search results
- `@olostep/google-maps`: Google Maps data

Use them with the `parser` parameter:

```python
scrape_website.invoke({
    "url": "https://www.amazon.com/dp/PRODUCT_ID",
    "parser": "@olostep/amazon-product"
})
```

## Pricing

Olostep offers competitive pricing:
- Pay only for what you use
- No hidden fees
- Volume discounts available

Visit https://olostep.com/pricing for details.

## Support

- Documentation: https://docs.olostep.com
- Issues: GitHub Issues
- Email: info@olostep.com

## Links

- PyPI: https://pypi.org/project/langchain-olostep/
- Documentation: https://docs.olostep.com/integrations/langchain
- Dashboard: https://olostep.com/dashboard
- Website: https://olostep.com

## License

MIT
