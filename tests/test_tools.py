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
