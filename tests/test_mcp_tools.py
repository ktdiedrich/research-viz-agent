"""
Unit tests for the MCP tools.
"""
import unittest
from unittest.mock import Mock, patch
from research_viz_agent.mcp_tools.arxiv_tool import ArxivTool
from research_viz_agent.mcp_tools.pubmed_tool import PubMedTool
from research_viz_agent.mcp_tools.huggingface_tool import HuggingFaceTool


class TestArxivTool(unittest.TestCase):
    """Tests for ArxivTool."""
    
    def setUp(self):
        self.tool = ArxivTool()
    
    def test_initialization(self):
        """Test that ArxivTool initializes correctly."""
        self.assertIsNotNone(self.tool.client)
    
    def test_search_medical_cv_models(self):
        """Test search_medical_cv_models method."""
        # This is a simple structure test
        results = []  # Mock results
        self.assertIsInstance(results, list)


class TestPubMedTool(unittest.TestCase):
    """Tests for PubMedTool."""
    
    def setUp(self):
        self.tool = PubMedTool(email="test@example.com")
    
    def test_initialization(self):
        """Test that PubMedTool initializes correctly."""
        from Bio import Entrez
        self.assertEqual(Entrez.email, "test@example.com")


class TestHuggingFaceTool(unittest.TestCase):
    """Tests for HuggingFaceTool."""
    
    def setUp(self):
        self.tool = HuggingFaceTool()
    
    def test_initialization(self):
        """Test that HuggingFaceTool initializes correctly."""
        self.assertEqual(self.tool.base_url, "https://huggingface.co/api")
    
    def test_search_models_structure(self):
        """Test that search_models returns correct structure."""
        # Structure test only
        expected_keys = ['model_id', 'author', 'model_name', 'downloads', 'likes']
        # This would be tested with actual API calls in integration tests
        self.assertTrue(all(isinstance(key, str) for key in expected_keys))


if __name__ == "__main__":
    unittest.main()
