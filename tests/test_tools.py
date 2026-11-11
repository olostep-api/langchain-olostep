"""
Tests for LangChain Olostep tools.
"""

import pytest
import json
from unittest.mock import Mock, patch
from langchain_olostep.tools import (
    scrape_website,
    scrape_batch,
    answer_question,
    search_web,
    get_batch_results,
    get_crawl_results,
    extract_urls,
    crawl_website,
    OlostepClient
)
from langchain_core.exceptions import LangChainException


class TestOlostepClient:
    """Test OlostepClient."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = OlostepClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client.base_url == "https://api.olostep.com/v1"
        assert client.headers["Authorization"] == "Bearer test-api-key"
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OLOSTEP_API_KEY environment variable is required"):
                OlostepClient()
    
    @patch('requests.post')
    def test_scrape_success(self, mock_post):
        """Test successful URL scraping."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "markdown_content": "# Test Content",
                "page_metadata": {"title": "Test"}
            },
            "retrieve_id": "test-id"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = OlostepClient("test-api-key")
        result = client.scrape("https://example.com")
        
        assert result["result"]["markdown_content"] == "# Test Content"
        mock_post.assert_called_once()


class TestScrapeWebsiteTool:
    """Test scrape_website tool."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_scrape_website_success(self, mock_client_class):
        """Test successful website scraping."""
        mock_client = Mock()
        mock_client.scrape.return_value = {
            "result": {
                "markdown_content": "# Test Content",
                "page_metadata": {"title": "Test"}
            },
            "retrieve_id": "test-id"
        }
        mock_client_class.return_value = mock_client
        
        result = scrape_website.invoke({
            "url": "https://example.com",
            "format": "markdown"
        })
        
        parsed = json.loads(result)
        assert "# Test Content" in parsed["content"]
        assert parsed["url"] == "https://example.com"
        assert parsed["scrape_id"] == "test-id"
    
    @patch('requests.post')
    def test_scrape_website_caps_content(self, mock_post):
        """Content longer than max_characters is cut and marked."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {"markdown_content": "x" * 25000, "markdown_hosted_url": "https://files.example/full.md"},
            "retrieve_id": "r1",
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        data = json.loads(scrape_website.invoke({"url": "https://example.com", "api_key": "test-key"}))
        assert len(data["content"]) == 20000
        assert data["truncated"] is True and data["full_length"] == 25000
        assert data["markdown_url"] == "https://files.example/full.md"

        data = json.loads(scrape_website.invoke({"url": "https://example.com", "max_characters": 30000, "api_key": "test-key"}))
        assert len(data["content"]) == 25000 and "truncated" not in data

    @patch('langchain_olostep.tools.OlostepClient')
    def test_scrape_website_error(self, mock_client_class):
        """Test scraping error handling."""
        mock_client = Mock()
        mock_client.scrape.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException):
            scrape_website.invoke({
                "url": "https://example.com",
                "format": "markdown"
            })


class TestScrapeBatchTool:
    """Test scrape_batch tool."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_scrape_batch_success(self, mock_client_class):
        """Test successful batch scraping."""
        mock_client = Mock()
        mock_client.batch_scrape.return_value = {
            "batch_id": "batch-123",
            "status": "in_progress"
        }
        mock_client_class.return_value = mock_client
        
        result = scrape_batch.invoke({
            "urls": ["https://example1.com", "https://example2.com"],
            "format": "markdown"
        })
        
        parsed = json.loads(result)
        assert parsed["batch_id"] == "batch-123"
        assert parsed["total_urls"] == 2
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_scrape_batch_error(self, mock_client_class):
        """Test batch scraping error handling."""
        mock_client = Mock()
        mock_client.batch_scrape.side_effect = Exception("Batch API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException):
            scrape_batch.invoke({
                "urls": ["https://example.com"],
                "format": "markdown"
            })


class TestAnswerQuestionTool:
    """Test answer_question tool."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_answer_question_success(self, mock_client_class):
        """Test successful question answering."""
        mock_client = Mock()
        mock_client.answer.return_value = {
            "result": {
                "json_content": '{"answer": "Paris"}',
                "sources": ["https://example.com"]
            },
            "id": "answer-123"
        }
        mock_client_class.return_value = mock_client
        
        result = answer_question.invoke({
            "task": "What is the capital of France?"
        })
        
        parsed = json.loads(result)
        assert parsed["answer"]["answer"] == "Paris"
        assert len(parsed["sources"]) == 1
        assert parsed["answer_id"] == "answer-123"
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_answer_question_with_schema(self, mock_client_class):
        """Test question answering with JSON schema."""
        mock_client = Mock()
        mock_client.answer.return_value = {
            "result": {
                "json_content": '{"city": "Paris", "country": "France"}',
                "sources": []
            },
            "id": "answer-456"
        }
        mock_client_class.return_value = mock_client
        
        result = answer_question.invoke({
            "task": "What is the capital of France?",
            "json_schema": {"city": "", "country": ""}
        })
        
        parsed = json.loads(result)
        assert parsed["answer"]["city"] == "Paris"
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_answer_question_error(self, mock_client_class):
        """Test answer question error handling."""
        mock_client = Mock()
        mock_client.answer.side_effect = Exception("Answer API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException):
            answer_question.invoke({
                "task": "What is the capital of France?"
            })


class TestSearchWebTool:
    """Test search_web tool."""
    
    @patch('requests.post')
    def test_search_web_sends_query_and_limit(self, mock_post):
        """The tool calls /v1/searches and returns one result per link."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "search_123",
            "result": {
                "links": [
                    {"url": "https://example.com/a", "title": "A", "description": "First"},
                    {"url": "https://example.com/b", "title": "B", "description": "Second"},
                ]
            },
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = search_web.invoke({"query": "test query", "api_key": "test-key"})
        
        assert mock_post.call_args.args[0] == "https://api.olostep.com/v1/searches"
        assert mock_post.call_args.kwargs["json"] == {"query": "test query", "limit": 5}
        data = json.loads(result)
        assert data["search_id"] == "search_123"
        assert data["total_results"] == 2
        assert data["results"][0] == {"url": "https://example.com/a", "title": "A", "description": "First"}
    
    @patch('requests.post')
    def test_search_web_with_filters_and_scrape(self, mock_post):
        """Domain filters and scrape_results are sent, and page markdown is returned."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "search_456",
            "result": {"links": [{"url": "https://docs.example.com", "title": "Docs", "description": "", "markdown_content": "# Docs"}]},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = search_web.invoke({
            "query": "rate limits",
            "limit": 100,
            "include_domains": ["docs.example.com"],
            "exclude_domains": ["pinterest.com"],
            "scrape_results": True,
            "api_key": "test-key",
        })
        
        assert mock_post.call_args.kwargs["json"] == {
            "query": "rate limits",
            "limit": 25,
            "include_domains": ["docs.example.com"],
            "exclude_domains": ["pinterest.com"],
            "scrape_options": {"formats": ["markdown"]},
        }
        assert json.loads(result)["results"][0]["markdown_content"] == "# Docs"
    
    @patch('requests.post')
    def test_search_web_caps_page_content(self, mock_post):
        """Scraped markdown longer than max_characters is cut and marked."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "search_789",
            "result": {"links": [
                {"url": "https://a.com", "title": "A", "description": "", "markdown_content": "x" * 25000},
                {"url": "https://b.com", "title": "B", "description": "", "markdown_content": "short"},
            ]},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        data = json.loads(search_web.invoke({"query": "q", "scrape_results": True, "api_key": "test-key"}))
        long_item, short_item = data["results"]
        assert len(long_item["markdown_content"]) == 20000
        assert long_item["truncated"] is True and long_item["full_length"] == 25000
        assert short_item["markdown_content"] == "short" and "truncated" not in short_item

        data = json.loads(search_web.invoke({"query": "q", "scrape_results": True, "max_characters": 100, "api_key": "test-key"}))
        assert len(data["results"][0]["markdown_content"]) == 100

    @patch('langchain_olostep.tools.OlostepClient')
    def test_search_web_error(self, mock_client_class):
        """Test search_web error handling."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException) as exc_info:
            search_web.invoke({"query": "test", "api_key": "test-key"})
        
        assert "Failed to search the web" in str(exc_info.value)


class TestExtractUrlsTool:
    """Test extract_urls tool."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_extract_urls_success(self, mock_client_class):
        """Test successful URL extraction."""
        mock_client = Mock()
        mock_client.create_map.return_value = {
            "urls": [
                "https://example.com/page1",
                "https://example.com/page2"
            ],
            "id": "map-123"
        }
        mock_client_class.return_value = mock_client
        
        result = extract_urls.invoke({
            "url": "https://example.com"
        })
        
        parsed = json.loads(result)
        assert parsed["total_urls"] == 2
        assert len(parsed["urls"]) == 2
        assert parsed["map_id"] == "map-123"
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_extract_urls_with_filters(self, mock_client_class):
        """Test URL extraction with filters."""
        mock_client = Mock()
        mock_client.create_map.return_value = {
            "urls": ["https://example.com/blog/post1"],
            "id": "map-456"
        }
        mock_client_class.return_value = mock_client
        
        result = extract_urls.invoke({
            "url": "https://example.com",
            "include_urls": ["/blog/**"],
            "top_n": 10
        })
        
        parsed = json.loads(result)
        assert parsed["total_urls"] == 1
        assert parsed["top_n"] == 10
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_extract_urls_error(self, mock_client_class):
        """Test URL extraction error handling."""
        mock_client = Mock()
        mock_client.create_map.side_effect = Exception("Map API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException):
            extract_urls.invoke({
                "url": "https://example.com"
            })


class TestCrawlWebsiteTool:
    """Test crawl_website tool."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_crawl_website_success(self, mock_client_class):
        """Test successful website crawling."""
        mock_client = Mock()
        mock_client.start_crawl.return_value = {
            "id": "crawl-123",
            "status": "in_progress",
            "pages_crawled": 5
        }
        mock_client_class.return_value = mock_client
        
        result = crawl_website.invoke({
            "start_url": "https://example.com",
            "max_pages": 10
        })
        
        parsed = json.loads(result)
        assert parsed["crawl_id"] == "crawl-123"
        assert parsed["status"] == "in_progress"
        assert parsed["max_pages"] == 10
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_crawl_website_with_filters(self, mock_client_class):
        """Test crawling with filters."""
        mock_client = Mock()
        mock_client.start_crawl.return_value = {
            "id": "crawl-456",
            "status": "in_progress",
            "pages_crawled": 0
        }
        mock_client_class.return_value = mock_client
        
        result = crawl_website.invoke({
            "start_url": "https://example.com",
            "max_pages": 50,
            "include_urls": ["/**"],
            "exclude_urls": ["/admin/**"],
            "max_depth": 3
        })
        
        parsed = json.loads(result)
        assert parsed["crawl_id"] == "crawl-456"
        assert parsed["max_depth"] == 3
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_crawl_website_error(self, mock_client_class):
        """Test crawling error handling."""
        mock_client = Mock()
        mock_client.start_crawl.side_effect = Exception("Crawl API Error")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(LangChainException):
            crawl_website.invoke({
                "start_url": "https://example.com",
                "max_pages": 10
            })


def _api_get(routes):
    """Fake requests.get that answers by URL path suffix."""
    def fake_get(url, headers=None, params=None, timeout=None):
        for suffix, body in routes.items():
            if url.endswith(suffix):
                response = Mock()
                response.json.return_value = body(params) if callable(body) else body
                response.raise_for_status = Mock()
                return response
        raise AssertionError(f"unexpected GET {url}")
    return fake_get


class TestJobResults:
    """Test get_batch_results and get_crawl_results."""

    @patch('requests.get')
    def test_batch_results_return_capped_content(self, mock_get):
        """Each item's content is fetched and cut to max_characters."""
        mock_get.side_effect = _api_get({
            "/batches/b1": {"id": "b1", "status": "completed", "total_urls": 2},
            "/batches/b1/items": {"items": [
                {"url": "https://a.com", "custom_id": "url_0", "retrieve_id": "r0"},
                {"url": "https://b.com", "custom_id": "url_1", "retrieve_id": "r1"},
            ], "cursor": 2},
            "/retrieve": lambda params: {"markdown_content": "x" * 25000 if ("retrieve_id", "r0") in params else "short"},
        })
        data = json.loads(get_batch_results.invoke({"batch_id": "b1", "api_key": "test-key"}))
        first, second = data["items"]
        assert len(first["content"]) == 20000 and first["truncated"] is True and first["full_length"] == 25000
        assert second == {"url": "https://b.com", "custom_id": "url_1", "content": "short"}
        # The first page held every URL, so there is nothing more even though the API sent a cursor.
        assert data["has_more"] is False and "next_cursor" not in data

    @patch('requests.get')
    def test_batch_results_page_through_cursor(self, mock_get):
        """has_more and next_cursor come back while more pages remain."""
        mock_get.side_effect = _api_get({
            "/batches/b1": {"id": "b1", "status": "completed", "total_urls": 3},
            "/batches/b1/items": {"items": [{"url": "https://a.com", "retrieve_id": "r0"}], "cursor": 99},
            "/retrieve": {"markdown_content": "page"},
        })
        data = json.loads(get_batch_results.invoke({"batch_id": "b1", "items_limit": 1, "api_key": "test-key"}))
        assert data["has_more"] is True and data["next_cursor"] == 99

    @patch('requests.get')
    def test_batch_still_running(self, mock_get):
        """A running batch returns its status and no items."""
        mock_get.side_effect = _api_get({"/batches/b1": {"id": "b1", "status": "in_progress"}})
        data = json.loads(get_batch_results.invoke({"batch_id": "b1", "api_key": "test-key"}))
        assert data["status"] == "in_progress" and "items" not in data

    @patch('requests.get')
    def test_crawl_results_return_capped_content(self, mock_get):
        """Crawl pages are fetched in the requested format and capped."""
        mock_get.side_effect = _api_get({
            "/crawls/c1": {"id": "c1", "status": "completed", "pages_count": 1},
            "/crawls/c1/pages": {"pages": [{"url": "https://docs.a.com", "retrieve_id": "r0"}], "cursor": None},
            "/retrieve": lambda params: {"html_content": "<p>" + "y" * 500 + "</p>"} if ("formats[]", "html") in params else {},
        })
        data = json.loads(get_crawl_results.invoke({"crawl_id": "c1", "format": "html", "max_characters": 100, "api_key": "test-key"}))
        assert data["crawl_id"] == "c1" and data["has_more"] is False
        assert len(data["items"][0]["content"]) == 100 and data["items"][0]["truncated"] is True

    @patch('requests.get')
    def test_missing_retrieve_id_is_reported(self, mock_get):
        """A page without a retrieve_id gets an error instead of failing the call."""
        mock_get.side_effect = _api_get({
            "/crawls/c1": {"id": "c1", "status": "completed"},
            "/crawls/c1/pages": {"pages": [{"url": "https://docs.a.com"}]},
        })
        data = json.loads(get_crawl_results.invoke({"crawl_id": "c1", "api_key": "test-key"}))
        assert "error" in data["items"][0]
