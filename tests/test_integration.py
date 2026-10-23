"""
Integration tests for LangChain Olostep package.
"""

import pytest
import json
from unittest.mock import Mock, patch
from langchain_olostep import (
    scrape_website,
    scrape_batch,
    answer_question,
    extract_urls,
    crawl_website
)


class TestIntegration:
    """Integration tests for the complete package."""
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_tool_imports(self, mock_client_class):
        """Test that all tools can be imported and used."""
        assert scrape_website is not None
        assert scrape_batch is not None
        assert answer_question is not None
        assert extract_urls is not None
        assert crawl_website is not None
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_tool_schemas(self, mock_client_class):
        """Test that tools have proper schemas."""
        mock_client = Mock()
        mock_client.scrape.return_value = {
            "result": {"markdown_content": "# Test"},
            "retrieve_id": "test-id"
        }
        mock_client_class.return_value = mock_client
        
        # Test tool schemas
        assert hasattr(scrape_website, 'args_schema')
        assert hasattr(scrape_batch, 'args_schema')
        assert hasattr(answer_question, 'args_schema')
        assert hasattr(extract_urls, 'args_schema')
        assert hasattr(crawl_website, 'args_schema')
        
        # Test that tools can be called
        result = scrape_website.invoke({
            "url": "https://example.com",
            "format": "markdown"
        })
        assert "Test" in result
    
    def test_package_version(self):
        """Test that package version is accessible."""
        from langchain_olostep import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
    
    def test_package_exports(self):
        """Test that package exports are correct."""
        from langchain_olostep import __all__
        
        expected_exports = [
            "scrape_website",
            "scrape_batch",
            "answer_question",
            "extract_urls",
            "crawl_website"
        ]
        
        for export in expected_exports:
            assert export in __all__
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_error_handling_consistency(self, mock_client_class):
        """Test that error handling is consistent across tools."""
        from langchain_core.exceptions import LangChainException
        
        # Mock client to raise errors
        mock_client = Mock()
        mock_client.scrape.side_effect = Exception("Test error")
        mock_client.batch_scrape.side_effect = Exception("Test error")
        mock_client.answer.side_effect = Exception("Test error")
        mock_client.create_map.side_effect = Exception("Test error")
        mock_client.start_crawl.side_effect = Exception("Test error")
        mock_client_class.return_value = mock_client
        
        # Test that all tools raise LangChainException
        with pytest.raises(LangChainException):
            scrape_website.invoke({
                "url": "https://example.com",
                "format": "markdown"
            })
        
        with pytest.raises(LangChainException):
            scrape_batch.invoke({
                "urls": ["https://example.com"],
                "format": "markdown"
            })
        
        with pytest.raises(LangChainException):
            answer_question.invoke({
                "task": "What is the capital of France?"
            })
        
        with pytest.raises(LangChainException):
            extract_urls.invoke({
                "url": "https://example.com"
            })
        
        with pytest.raises(LangChainException):
            crawl_website.invoke({
                "start_url": "https://example.com",
                "max_pages": 10
            })
    
    @patch('langchain_olostep.tools.OlostepClient')
    def test_format_consistency(self, mock_client_class):
        """Test that format handling is consistent across tools."""
        mock_client = Mock()
        mock_client.scrape.return_value = {
            "result": {
                "markdown_content": "# Markdown Content",
                "html_content": "<h1>HTML Content</h1>",
                "text_content": "Plain Text Content",
                "json_content": {"content": "JSON Content"},
                "page_metadata": {}
            },
            "retrieve_id": "test-id"
        }
        mock_client_class.return_value = mock_client
        
        # Test different formats
        formats = ["markdown", "html", "text", "json"]
        for format_type in formats:
            result = scrape_website.invoke({
                "url": "https://example.com",
                "format": format_type
            })
            
            parsed = json.loads(result)
            assert "content" in parsed
            assert parsed["format"] == format_type
    
    def test_environment_variable_handling(self):
        """Test that environment variable handling works correctly."""
        import os
        
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OLOSTEP_API_KEY environment variable is required"):
                from langchain_olostep.tools import OlostepClient
                OlostepClient()
        
        # Test with API key in environment
        with patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-key"}):
            from langchain_olostep.tools import OlostepClient
            client = OlostepClient()
            assert client.api_key == "test-key"
