"""
Unit tests for ArXiv MCP tool.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
import arxiv

from research_viz_agent.mcp_tools.arxiv_tool import (
    ArxivTool,
    create_arxiv_tool,
    get_arxiv_server,
    search_arxiv_papers,
    search_medical_cv_papers
)


class TestArxivToolInitialization:
    """Tests for ArxivTool initialization."""
    
    def test_init_creates_client(self):
        """Test that initialization creates an arxiv client."""
        tool = ArxivTool()
        assert hasattr(tool, 'client')
        assert isinstance(tool.client, arxiv.Client)
    
    def test_create_arxiv_tool_factory(self):
        """Test the factory function creates an ArxivTool instance."""
        tool = create_arxiv_tool()
        assert isinstance(tool, ArxivTool)
    
    def test_get_arxiv_server(self):
        """Test getting the MCP server instance."""
        from mcp.server import Server
        server = get_arxiv_server()
        assert isinstance(server, Server)


class TestArxivToolSearchPapers:
    """Tests for ArxivTool.search_papers method."""
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_search_papers_basic(self, mock_client_class):
        """Test basic paper search functionality."""
        # Create mock author objects with name property
        mock_author1 = MagicMock()
        mock_author1.name = "Author One"
        mock_author2 = MagicMock()
        mock_author2.name = "Author Two"
        
        # Create mock paper
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.authors = [mock_author1, mock_author2]
        mock_paper.summary = "This is a test summary"
        mock_paper.published = datetime(2024, 1, 1)
        mock_paper.updated = datetime(2024, 1, 2)
        mock_paper.categories = ["cs.CV", "cs.AI"]
        mock_paper.pdf_url = "http://arxiv.org/pdf/1234.5678"
        mock_paper.entry_id = "http://arxiv.org/abs/1234.5678"
        mock_paper.doi = "10.1234/test"
        mock_paper.primary_category = "cs.CV"
        
        # Mock client results
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_paper]
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        results = tool.search_papers("machine learning", max_results=5)
        
        assert len(results) == 1
        assert results[0]['title'] == "Test Paper"
        assert len(results[0]['authors']) == 2
        assert results[0]['authors'][0] == "Author One"
        assert results[0]['summary'] == "This is a test summary"
        assert results[0]['pdf_url'] == "http://arxiv.org/pdf/1234.5678"
        assert results[0]['categories'] == ["cs.CV", "cs.AI"]
        assert results[0]['primary_category'] == "cs.CV"
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_search_papers_multiple_results(self, mock_client_class):
        """Test search with multiple paper results."""
        mock_papers = []
        for i in range(3):
            mock_paper = MagicMock()
            mock_paper.title = f"Paper {i}"
            mock_paper.authors = [MagicMock(name=f"Author {i}")]
            mock_paper.summary = f"Summary {i}"
            mock_paper.published = datetime(2024, 1, i+1)
            mock_paper.updated = datetime(2024, 1, i+1)
            mock_paper.categories = ["cs.CV"]
            mock_paper.pdf_url = f"http://arxiv.org/pdf/{i}"
            mock_paper.entry_id = f"http://arxiv.org/abs/{i}"
            mock_paper.doi = None
            mock_paper.primary_category = "cs.CV"
            mock_papers.append(mock_paper)
        
        mock_client = MagicMock()
        mock_client.results.return_value = mock_papers
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        results = tool.search_papers("test query", max_results=10)
        
        assert len(results) == 3
        assert results[0]['title'] == "Paper 0"
        assert results[1]['title'] == "Paper 1"
        assert results[2]['title'] == "Paper 2"
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_search_papers_empty_results(self, mock_client_class):
        """Test search with no results."""
        mock_client = MagicMock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        results = tool.search_papers("nonexistent query")
        
        assert len(results) == 0
        assert results == []
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_search_papers_with_none_dates(self, mock_client_class):
        """Test search papers with None for published/updated dates."""
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper"
        mock_paper.authors = [MagicMock(name="Author")]
        mock_paper.summary = "Summary"
        mock_paper.published = None
        mock_paper.updated = None
        mock_paper.categories = ["cs.CV"]
        mock_paper.pdf_url = "http://arxiv.org/pdf/1234"
        mock_paper.entry_id = "http://arxiv.org/abs/1234"
        mock_paper.doi = None
        mock_paper.primary_category = "cs.CV"
        
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_paper]
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        results = tool.search_papers("test")
        
        assert results[0]['published'] is None
        assert results[0]['updated'] is None
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_search_papers_custom_sort(self, mock_client_class):
        """Test search with custom sort criterion."""
        mock_client = MagicMock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        tool.search_papers("test", sort_by=arxiv.SortCriterion.SubmittedDate)
        
        # Verify Search was called with correct sort criterion
        mock_client.results.assert_called_once()


class TestArxivToolSearchMedicalCV:
    """Tests for ArxivTool.search_medical_cv_models method."""
    
    @patch.object(ArxivTool, 'search_papers')
    def test_search_medical_cv_models_basic(self, mock_search):
        """Test basic medical CV search."""
        mock_search.return_value = [{'title': 'Medical CV Paper'}]
        
        tool = ArxivTool()
        results = tool.search_medical_cv_models()
        
        # Verify search_papers was called with medical CV query
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert 'cs.CV' in args[0]
        assert 'medical' in args[0].lower()
        assert kwargs['max_results'] == 20
    
    @patch.object(ArxivTool, 'search_papers')
    def test_search_medical_cv_models_with_additional_terms(self, mock_search):
        """Test medical CV search with additional terms."""
        mock_search.return_value = []
        
        tool = ArxivTool()
        tool.search_medical_cv_models(additional_terms="lung cancer", max_results=10)
        
        args, kwargs = mock_search.call_args
        query = args[0]
        assert 'cs.CV' in query
        assert 'medical' in query.lower()
        assert 'lung cancer' in query
        assert kwargs['max_results'] == 10
    
    @patch.object(ArxivTool, 'search_papers')
    def test_search_medical_cv_models_contains_medical_terms(self, mock_search):
        """Test that medical CV query contains expected medical terms."""
        mock_search.return_value = []
        
        tool = ArxivTool()
        tool.search_medical_cv_models()
        
        args, _ = mock_search.call_args
        query = args[0].lower()
        
        # Check for expected medical terms
        assert any(term in query for term in ['medical', 'clinical', 'radiology', 'pathology'])
    
    @patch.object(ArxivTool, 'search_papers')
    def test_search_medical_cv_models_custom_max_results(self, mock_search):
        """Test medical CV search with custom max results."""
        mock_search.return_value = []
        
        tool = ArxivTool()
        tool.search_medical_cv_models(max_results=50)
        
        _, kwargs = mock_search.call_args
        assert kwargs['max_results'] == 50


class TestSearchArxivPapersAsync:
    """Tests for search_arxiv_papers async function."""
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_arxiv_papers_success(self, mock_tool):
        """Test successful arXiv paper search."""
        mock_tool.search_papers.return_value = [
            {
                'title': 'Test Paper',
                'authors': ['Author One', 'Author Two'],
                'published': '2024-01-01',
                'categories': ['cs.CV'],
                'summary': 'A' * 400,  # Long summary to test truncation
                'pdf_url': 'http://arxiv.org/pdf/1234'
            }
        ]
        
        arguments = {'query': 'machine learning', 'max_results': 5}
        results = await search_arxiv_papers(arguments)
        
        assert len(results) == 1
        assert results[0].type == "text"
        assert 'Test Paper' in results[0].text
        assert 'Found 1 papers' in results[0].text
        mock_tool.search_papers.assert_called_once_with('machine learning', 5)
    
    @pytest.mark.asyncio
    async def test_search_arxiv_papers_missing_query(self):
        """Test arXiv search with missing query parameter."""
        arguments = {}
        results = await search_arxiv_papers(arguments)
        
        assert len(results) == 1
        assert results[0].type == "text"
        assert 'Error' in results[0].text
        assert 'required' in results[0].text
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_arxiv_papers_default_max_results(self, mock_tool):
        """Test arXiv search uses default max_results."""
        mock_tool.search_papers.return_value = []
        
        arguments = {'query': 'test'}
        await search_arxiv_papers(arguments)
        
        mock_tool.search_papers.assert_called_once_with('test', 10)
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_arxiv_papers_error_handling(self, mock_tool):
        """Test error handling in arXiv search."""
        mock_tool.search_papers.side_effect = Exception("Search failed")
        
        arguments = {'query': 'test query'}
        results = await search_arxiv_papers(arguments)
        
        assert len(results) == 1
        assert results[0].type == "text"
        assert 'Error searching arXiv' in results[0].text
        assert 'Search failed' in results[0].text
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_arxiv_papers_multiple_authors(self, mock_tool):
        """Test formatting with multiple authors (truncation)."""
        mock_tool.search_papers.return_value = [
            {
                'title': 'Paper',
                'authors': ['Author 1', 'Author 2', 'Author 3', 'Author 4', 'Author 5'],
                'published': '2024-01-01',
                'categories': ['cs.CV'],
                'summary': 'Short summary',
                'pdf_url': 'http://test.com'
            }
        ]
        
        arguments = {'query': 'test'}
        results = await search_arxiv_papers(arguments)
        
        assert '...' in results[0].text  # Should show truncation for authors
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_arxiv_papers_summary_truncation(self, mock_tool):
        """Test that long summaries are truncated."""
        long_summary = 'A' * 500
        mock_tool.search_papers.return_value = [
            {
                'title': 'Paper',
                'authors': ['Author'],
                'published': '2024-01-01',
                'categories': ['cs.CV'],
                'summary': long_summary,
                'pdf_url': 'http://test.com'
            }
        ]
        
        arguments = {'query': 'test'}
        results = await search_arxiv_papers(arguments)
        
        # Summary should be truncated to 300 chars
        assert long_summary[:300] in results[0].text
        assert 'Summary:' in results[0].text


class TestSearchMedicalCVPapersAsync:
    """Tests for search_medical_cv_papers async function."""
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_medical_cv_papers_success(self, mock_tool):
        """Test successful medical CV paper search."""
        mock_tool.search_medical_cv_models.return_value = [
            {
                'title': 'Medical CV Paper',
                'authors': ['Medical Author'],
                'published': '2024-01-01',
                'categories': ['cs.CV', 'q-bio'],
                'summary': 'Medical computer vision summary',
                'pdf_url': 'http://arxiv.org/pdf/5678'
            }
        ]
        
        arguments = {'additional_terms': 'lung cancer'}
        results = await search_medical_cv_papers(arguments)
        
        assert len(results) == 1
        assert results[0].type == "text"
        assert 'Medical CV Paper' in results[0].text
        assert 'Found 1 medical CV papers' in results[0].text
        mock_tool.search_medical_cv_models.assert_called_once_with('lung cancer')
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_medical_cv_papers_no_additional_terms(self, mock_tool):
        """Test medical CV search without additional terms."""
        mock_tool.search_medical_cv_models.return_value = []
        
        arguments = {}
        await search_medical_cv_papers(arguments)
        
        mock_tool.search_medical_cv_models.assert_called_once_with('')
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_medical_cv_papers_error_handling(self, mock_tool):
        """Test error handling in medical CV search."""
        mock_tool.search_medical_cv_models.side_effect = Exception("Medical search failed")
        
        arguments = {}
        results = await search_medical_cv_papers(arguments)
        
        assert len(results) == 1
        assert results[0].type == "text"
        assert 'Error searching medical CV papers' in results[0].text
        assert 'Medical search failed' in results[0].text
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv_tool')
    async def test_search_medical_cv_papers_formatting(self, mock_tool):
        """Test proper formatting of medical CV papers."""
        mock_tool.search_medical_cv_models.return_value = [
            {
                'title': 'Paper 1',
                'authors': ['A1', 'A2'],
                'published': '2024-01-01',
                'categories': ['cs.CV'],
                'summary': 'Summary 1',
                'pdf_url': 'http://url1.com'
            },
            {
                'title': 'Paper 2',
                'authors': ['B1'],
                'published': '2024-02-01',
                'categories': ['cs.AI'],
                'summary': 'Summary 2',
                'pdf_url': 'http://url2.com'
            }
        ]
        
        arguments = {'additional_terms': 'test'}
        results = await search_medical_cv_papers(arguments)
        
        text = results[0].text
        assert 'Paper 1' in text
        assert 'Paper 2' in text
        assert 'http://url1.com' in text
        assert 'http://url2.com' in text
        assert 'Found 2 medical CV papers' in text


class TestArxivToolIntegration:
    """Integration tests for ArxivTool."""
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_full_workflow_search_to_results(self, mock_client_class):
        """Test complete workflow from search to formatted results."""
        # Setup mock
        mock_paper = MagicMock()
        mock_paper.title = "Integration Test Paper"
        mock_paper.authors = [MagicMock(name="Test Author")]
        mock_paper.summary = "This is an integration test"
        mock_paper.published = datetime(2024, 1, 1)
        mock_paper.updated = datetime(2024, 1, 1)
        mock_paper.categories = ["cs.CV", "cs.LG"]
        mock_paper.pdf_url = "http://arxiv.org/pdf/test"
        mock_paper.entry_id = "http://arxiv.org/abs/test"
        mock_paper.doi = None
        mock_paper.primary_category = "cs.CV"
        
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_paper]
        mock_client_class.return_value = mock_client
        
        # Create tool and search
        tool = create_arxiv_tool()
        results = tool.search_papers("test query", max_results=5)
        
        # Verify results structure
        assert len(results) == 1
        assert 'title' in results[0]
        assert 'authors' in results[0]
        assert 'summary' in results[0]
        assert 'pdf_url' in results[0]
        assert 'categories' in results[0]
        assert 'published' in results[0]
        assert 'entry_id' in results[0]
    
    @patch('research_viz_agent.mcp_tools.arxiv_tool.arxiv.Client')
    def test_medical_cv_search_uses_correct_base_query(self, mock_client_class):
        """Test that medical CV search constructs proper query."""
        mock_client = MagicMock()
        mock_client.results.return_value = []
        mock_client_class.return_value = mock_client
        
        tool = ArxivTool()
        tool.search_medical_cv_models(additional_terms="lung detection")
        
        # The search should have been called with a query containing both base and additional terms
        mock_client.results.assert_called_once()
