"""
Unit tests for the PubMed MCP tool.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock, mock_open
import mcp.types as types

from research_viz_agent.mcp_tools.pubmed_tool import (
    PubMedTool,
    create_pubmed_tool,
    get_pubmed_server,
    search_pubmed_papers,
    search_medical_cv_papers_pubmed,
    get_paper_by_pmid
)


class TestPubMedToolInitialization:
    """Test PubMedTool initialization."""
    
    def test_init_with_default_email(self):
        """Test initialization with default email."""
        with patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez') as mock_entrez:
            PubMedTool()
            assert mock_entrez.email == "research@example.com"
    
    def test_init_with_custom_email(self):
        """Test initialization with custom email."""
        with patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez') as mock_entrez:
            PubMedTool(email="custom@test.com")
            assert mock_entrez.email == "custom@test.com"
    
    def test_create_pubmed_tool_factory(self):
        """Test factory function creates PubMedTool instance."""
        with patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez'):
            tool = create_pubmed_tool(email="factory@test.com")
            assert isinstance(tool, PubMedTool)
    
    def test_get_pubmed_server(self):
        """Test server instance retrieval."""
        from mcp.server import Server
        server = get_pubmed_server()
        assert isinstance(server, Server)


class TestPubMedToolSearchPapers:
    """Test search_papers functionality."""
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.time.sleep')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Medline.parse')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.efetch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_basic(self, mock_entrez, mock_esearch, mock_read, 
                                  mock_efetch, mock_medline_parse, mock_sleep):
        """Test basic paper search functionality."""
        # Mock esearch response
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        mock_read.return_value = {"IdList": ["12345678"]}
        
        # Mock efetch and medline parse
        mock_fetch_handle = MagicMock()
        mock_efetch.return_value = mock_fetch_handle
        
        mock_paper = {
            'PMID': '12345678',
            'TI': 'Test Medical Paper',
            'AU': ['Smith J', 'Doe A'],
            'AB': 'This is a test abstract about medical imaging.',
            'TA': 'Medical Imaging Journal',
            'DP': '2024 Jan',
            'OT': ['medical imaging', 'AI'],
            'MH': ['Radiology', 'Artificial Intelligence'],
            'LID': '10.1234/test'
        }
        mock_medline_parse.return_value = [mock_paper]
        
        tool = PubMedTool()
        results = tool.search_papers("medical imaging AI", max_results=5)
        
        assert len(results) == 1
        assert results[0]['pmid'] == '12345678'
        assert results[0]['title'] == 'Test Medical Paper'
        assert len(results[0]['authors']) == 2
        assert results[0]['authors'][0] == 'Smith J'
        assert results[0]['journal'] == 'Medical Imaging Journal'
        
        # Verify API calls
        mock_esearch.assert_called_once_with(
            db="pubmed",
            term="medical imaging AI",
            retmax=5,
            sort="relevance"
        )
        mock_efetch.assert_called_once()
        mock_sleep.assert_called_once_with(0.34)
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_empty_results(self, mock_entrez, mock_esearch, mock_read):
        """Test search with no results."""
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        mock_read.return_value = {"IdList": []}
        
        tool = PubMedTool()
        results = tool.search_papers("nonexistent query", max_results=10)
        
        assert len(results) == 0
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.time.sleep')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Medline.parse')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.efetch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_multiple_results(self, mock_entrez, mock_esearch, mock_read,
                                           mock_efetch, mock_medline_parse, mock_sleep):
        """Test search returning multiple papers."""
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        mock_read.return_value = {"IdList": ["111", "222", "333"]}
        
        mock_fetch_handle = MagicMock()
        mock_efetch.return_value = mock_fetch_handle
        
        mock_papers = [
            {
                'PMID': f'{i}',
                'TI': f'Paper {i}',
                'AU': ['Author'],
                'AB': 'Abstract',
                'TA': 'Journal',
                'DP': '2024',
                'OT': [],
                'MH': [],
                'LID': ''
            }
            for i in [111, 222, 333]
        ]
        mock_medline_parse.return_value = mock_papers
        
        tool = PubMedTool()
        results = tool.search_papers("test", max_results=3)
        
        assert len(results) == 3
        assert results[0]['pmid'] == '111'
        assert results[2]['pmid'] == '333'
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.time.sleep')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Medline.parse')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.efetch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_custom_sort(self, mock_entrez, mock_esearch, mock_read,
                                      mock_efetch, mock_medline_parse, mock_sleep):
        """Test search with custom sort order."""
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        mock_read.return_value = {"IdList": ["123"]}
        
        mock_fetch_handle = MagicMock()
        mock_efetch.return_value = mock_fetch_handle
        
        mock_medline_parse.return_value = [{
            'PMID': '123',
            'TI': 'Test',
            'AU': [],
            'AB': '',
            'TA': '',
            'DP': '',
            'OT': [],
            'MH': [],
            'LID': ''
        }]
        
        tool = PubMedTool()
        tool.search_papers("test", sort="pub_date")
        
        mock_esearch.assert_called_once()
        call_kwargs = mock_esearch.call_args[1]
        assert call_kwargs['sort'] == 'pub_date'
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_error_handling(self, mock_entrez, mock_esearch):
        """Test error handling in search."""
        mock_esearch.side_effect = Exception("API Error")
        
        tool = PubMedTool()
        results = tool.search_papers("test")
        
        assert results == []
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.time.sleep')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Medline.parse')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.efetch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_papers_missing_fields(self, mock_entrez, mock_esearch, mock_read,
                                         mock_efetch, mock_medline_parse, mock_sleep):
        """Test handling of papers with missing fields."""
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        mock_read.return_value = {"IdList": ["999"]}
        
        mock_fetch_handle = MagicMock()
        mock_efetch.return_value = mock_fetch_handle
        
        # Paper with minimal fields
        mock_medline_parse.return_value = [{'PMID': '999'}]
        
        tool = PubMedTool()
        results = tool.search_papers("test", max_results=1)
        
        assert len(results) == 1
        assert results[0]['pmid'] == '999'
        assert results[0]['title'] == ''
        assert results[0]['authors'] == []
        assert results[0]['abstract'] == ''


class TestPubMedToolSearchMedicalCV:
    """Test search_medical_cv_models functionality."""
    
    @patch.object(PubMedTool, 'search_papers')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_medical_cv_basic(self, mock_entrez, mock_search):
        """Test basic medical CV model search."""
        mock_search.return_value = [
            {
                'pmid': '11111',
                'title': 'Deep Learning for Medical Imaging',
                'authors': ['Researcher A'],
                'abstract': 'Study on AI in radiology',
                'journal': 'AI in Medicine',
                'publication_date': '2024',
                'keywords': [],
                'mesh_terms': [],
                'doi': ''
            }
        ]
        
        tool = PubMedTool()
        results = tool.search_medical_cv_models()
        
        assert len(results) == 1
        assert results[0]['pmid'] == '11111'
        
        # Verify the search query includes medical CV terms
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        query = call_args[0][0]
        assert 'computer vision' in query
        assert 'medical imaging' in query
    
    @patch.object(PubMedTool, 'search_papers')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_medical_cv_with_additional_terms(self, mock_entrez, mock_search):
        """Test medical CV search with additional terms."""
        mock_search.return_value = []
        
        tool = PubMedTool()
        tool.search_medical_cv_models(additional_terms="chest xray", max_results=15)
        
        call_args = mock_search.call_args
        query = call_args[0][0]
        assert 'chest xray' in query
        assert call_args[1]['max_results'] == 15
    
    @patch.object(PubMedTool, 'search_papers')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_medical_cv_constructs_correct_query(self, mock_entrez, mock_search):
        """Test that medical CV search builds the correct base query."""
        mock_search.return_value = []
        
        tool = PubMedTool()
        tool.search_medical_cv_models()
        
        call_args = mock_search.call_args
        query = call_args[0][0]
        
        # Check for key terms in base query
        assert 'computer vision' in query.lower()
        assert 'deep learning' in query.lower()
        assert 'convolutional neural network' in query.lower()
        assert 'radiology' in query.lower()
        assert 'pathology' in query.lower()
    
    @patch.object(PubMedTool, 'search_papers')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_search_medical_cv_custom_max_results(self, mock_entrez, mock_search):
        """Test medical CV search with custom max results."""
        mock_search.return_value = []
        
        tool = PubMedTool()
        tool.search_medical_cv_models(max_results=30)
        
        call_args = mock_search.call_args
        assert call_args[1]['max_results'] == 30


class TestSearchPubMedPapersAsync:
    """Test async search_pubmed_papers MCP function."""
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_success(self, mock_tool):
        """Test successful async paper search."""
        mock_tool.search_papers.return_value = [
            {
                'pmid': '12345',
                'title': 'Medical AI Study',
                'authors': ['Smith J', 'Doe A', 'Johnson B'],
                'abstract': 'Abstract text here',
                'journal': 'Medical Journal',
                'publication_date': '2024 Jan',
                'keywords': ['AI', 'medical'],
                'mesh_terms': ['Radiology', 'Machine Learning'],
                'doi': '10.1234/test'
            }
        ]
        
        result = await search_pubmed_papers({
            'query': 'medical AI',
            'max_results': 10,
            'sort': 'relevance'
        })
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert 'Medical AI Study' in result[0].text
        assert 'Found 1 PubMed papers' in result[0].text
        assert '12345' in result[0].text
        
        mock_tool.search_papers.assert_called_once_with('medical AI', 10, 'relevance')
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_missing_query(self, mock_tool):
        """Test async search with missing query."""
        result = await search_pubmed_papers({})
        
        assert len(result) == 1
        assert 'Error: Query parameter is required' in result[0].text
        mock_tool.search_papers.assert_not_called()
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_no_results(self, mock_tool):
        """Test async search with no results."""
        mock_tool.search_papers.return_value = []
        
        result = await search_pubmed_papers({'query': 'nonexistent'})
        
        assert len(result) == 1
        assert 'No papers found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_default_params(self, mock_tool):
        """Test async search with default parameters."""
        mock_tool.search_papers.return_value = []
        
        await search_pubmed_papers({'query': 'test'})
        
        mock_tool.search_papers.assert_called_once_with('test', 10, 'relevance')
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_error_handling(self, mock_tool):
        """Test error handling in async search."""
        mock_tool.search_papers.side_effect = Exception("Search failed")
        
        result = await search_pubmed_papers({'query': 'test'})
        
        assert len(result) == 1
        assert 'Error searching PubMed' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_author_truncation(self, mock_tool):
        """Test author list truncation in results."""
        mock_tool.search_papers.return_value = [
            {
                'pmid': '999',
                'title': 'Test Paper',
                'authors': ['Author1', 'Author2', 'Author3', 'Author4', 'Author5'],
                'abstract': 'Short abstract',
                'journal': 'Journal',
                'publication_date': '2024',
                'keywords': [],
                'mesh_terms': [],
                'doi': ''
            }
        ]
        
        result = await search_pubmed_papers({'query': 'test'})
        
        text = result[0].text
        assert 'et al.' in text
        assert '5 total' in text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_abstract_truncation(self, mock_tool):
        """Test abstract truncation in results."""
        long_abstract = 'A' * 500
        mock_tool.search_papers.return_value = [
            {
                'pmid': '888',
                'title': 'Test',
                'authors': [],
                'abstract': long_abstract,
                'journal': '',
                'publication_date': '',
                'keywords': [],
                'mesh_terms': [],
                'doi': ''
            }
        ]
        
        result = await search_pubmed_papers({'query': 'test'})
        
        text = result[0].text
        assert '...' in text
        # Abstract should be truncated
        assert text.count('A') < 500
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_pubmed_papers_mesh_truncation(self, mock_tool):
        """Test MeSH terms truncation in results."""
        mock_tool.search_papers.return_value = [
            {
                'pmid': '777',
                'title': 'Test',
                'authors': [],
                'abstract': '',
                'journal': '',
                'publication_date': '',
                'keywords': [],
                'mesh_terms': ['Term1', 'Term2', 'Term3', 'Term4', 'Term5', 'Term6'],
                'doi': ''
            }
        ]
        
        result = await search_pubmed_papers({'query': 'test'})
        
        text = result[0].text
        # Should show first 5 MeSH terms with ellipsis
        assert 'Term1' in text
        assert 'Term5' in text


class TestSearchMedicalCVPapersPubMedAsync:
    """Test async search_medical_cv_papers_pubmed MCP function."""
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_medical_cv_papers_success(self, mock_tool):
        """Test successful async medical CV paper search."""
        mock_tool.search_medical_cv_models.return_value = [
            {
                'pmid': '54321',
                'title': 'Deep Learning in Radiology',
                'authors': ['Researcher X'],
                'abstract': 'Study on CNN for medical imaging',
                'journal': 'Radiology AI',
                'publication_date': '2024',
                'keywords': [],
                'mesh_terms': ['Deep Learning', 'Radiology'],
                'doi': '10.5678/test'
            }
        ]
        
        result = await search_medical_cv_papers_pubmed({
            'additional_terms': 'chest CT'
        })
        
        assert len(result) == 1
        assert 'Found 1 medical CV papers' in result[0].text
        assert 'Deep Learning in Radiology' in result[0].text
        
        mock_tool.search_medical_cv_models.assert_called_once_with('chest CT')
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_medical_cv_papers_no_results(self, mock_tool):
        """Test async medical CV search with no results."""
        mock_tool.search_medical_cv_models.return_value = []
        
        result = await search_medical_cv_papers_pubmed({})
        
        assert len(result) == 1
        assert 'No medical computer vision papers found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_medical_cv_papers_default_params(self, mock_tool):
        """Test async medical CV search with default parameters."""
        mock_tool.search_medical_cv_models.return_value = []
        
        await search_medical_cv_papers_pubmed({})
        
        mock_tool.search_medical_cv_models.assert_called_once_with('')
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_search_medical_cv_papers_error_handling(self, mock_tool):
        """Test error handling in async medical CV search."""
        mock_tool.search_medical_cv_models.side_effect = Exception("Search error")
        
        result = await search_medical_cv_papers_pubmed({})
        
        assert len(result) == 1
        assert 'Error searching medical CV papers' in result[0].text


class TestGetPaperByPMIDAsync:
    """Test async get_paper_by_pmid MCP function."""
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_get_paper_by_pmid_success(self, mock_tool):
        """Test successful async paper retrieval by PMID."""
        mock_tool.search_papers.return_value = [
            {
                'pmid': '12345678',
                'title': 'Specific Medical Paper',
                'authors': ['Author A', 'Author B'],
                'abstract': 'Full abstract text here',
                'journal': 'Journal Name',
                'publication_date': '2024 Jan 15',
                'keywords': ['keyword1', 'keyword2'],
                'mesh_terms': ['MeSH1', 'MeSH2'],
                'doi': '10.1111/journal.12345'
            }
        ]
        
        result = await get_paper_by_pmid({'pmid': '12345678'})
        
        assert len(result) == 1
        text = result[0].text
        assert 'PMID: 12345678' in text
        assert 'Specific Medical Paper' in text
        assert 'Author A, Author B' in text
        assert 'Full abstract text here' in text
        
        mock_tool.search_papers.assert_called_once_with('12345678[PMID]', max_results=1)
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_get_paper_by_pmid_missing_pmid(self, mock_tool):
        """Test async paper retrieval with missing PMID."""
        result = await get_paper_by_pmid({})
        
        assert len(result) == 1
        assert 'Error: PMID parameter is required' in result[0].text
        mock_tool.search_papers.assert_not_called()
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_get_paper_by_pmid_not_found(self, mock_tool):
        """Test async paper retrieval when PMID not found."""
        mock_tool.search_papers.return_value = []
        
        result = await get_paper_by_pmid({'pmid': '99999999'})
        
        assert len(result) == 1
        assert 'not found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_get_paper_by_pmid_error_handling(self, mock_tool):
        """Test error handling in async paper retrieval."""
        mock_tool.search_papers.side_effect = Exception("Retrieval error")
        
        result = await get_paper_by_pmid({'pmid': '12345'})
        
        assert len(result) == 1
        assert 'Error retrieving paper details' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.pubmed_tool')
    async def test_get_paper_by_pmid_empty_fields(self, mock_tool):
        """Test paper retrieval with empty optional fields."""
        mock_tool.search_papers.return_value = [
            {
                'pmid': '11111',
                'title': 'Test Paper',
                'authors': [],
                'abstract': '',
                'journal': '',
                'publication_date': '',
                'keywords': [],
                'mesh_terms': [],
                'doi': ''
            }
        ]
        
        result = await get_paper_by_pmid({'pmid': '11111'})
        
        text = result[0].text
        assert 'Test Paper' in text
        assert 'N/A' in text  # Empty fields should show N/A


class TestPubMedToolIntegration:
    """Test integration scenarios."""
    
    @patch('research_viz_agent.mcp_tools.pubmed_tool.time.sleep')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Medline.parse')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.efetch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.read')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez.esearch')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_full_workflow_search_to_pmid_lookup(self, mock_entrez, mock_esearch, 
                                                 mock_read, mock_efetch, 
                                                 mock_medline_parse, mock_sleep):
        """Test complete workflow from search to PMID lookup."""
        # First search
        mock_search_handle = MagicMock()
        mock_esearch.return_value = mock_search_handle
        
        # First call returns multiple IDs, second call returns specific PMID
        mock_read.side_effect = [
            {"IdList": ["111", "222", "333"]},
            {"IdList": ["222"]}
        ]
        
        mock_fetch_handle = MagicMock()
        mock_efetch.return_value = mock_fetch_handle
        
        # First search returns multiple papers
        mock_papers_search = [
            {
                'PMID': '111',
                'TI': 'Paper 1',
                'AU': [],
                'AB': '',
                'TA': '',
                'DP': '',
                'OT': [],
                'MH': [],
                'LID': ''
            },
            {
                'PMID': '222',
                'TI': 'Target Paper',
                'AU': ['Author X'],
                'AB': 'Detailed abstract',
                'TA': 'Journal',
                'DP': '2024',
                'OT': [],
                'MH': [],
                'LID': ''
            },
            {
                'PMID': '333',
                'TI': 'Paper 3',
                'AU': [],
                'AB': '',
                'TA': '',
                'DP': '',
                'OT': [],
                'MH': [],
                'LID': ''
            }
        ]
        
        # Second search returns just the target paper
        mock_paper_detail = [{
            'PMID': '222',
            'TI': 'Target Paper',
            'AU': ['Author X'],
            'AB': 'Detailed abstract',
            'TA': 'Journal',
            'DP': '2024',
            'OT': [],
            'MH': [],
            'LID': ''
        }]
        
        mock_medline_parse.side_effect = [mock_papers_search, mock_paper_detail]
        
        tool = PubMedTool()
        
        # First search
        results = tool.search_papers("test query", max_results=3)
        assert len(results) == 3
        target_pmid = results[1]['pmid']
        
        # Then lookup specific paper
        detail = tool.search_papers(f"{target_pmid}[PMID]", max_results=1)
        assert len(detail) == 1
        assert detail[0]['pmid'] == '222'
        assert detail[0]['title'] == 'Target Paper'
    
    @patch.object(PubMedTool, 'search_papers')
    @patch('research_viz_agent.mcp_tools.pubmed_tool.Entrez')
    def test_medical_cv_search_builds_comprehensive_query(self, mock_entrez, mock_search):
        """Test that medical CV search constructs a comprehensive query."""
        mock_search.return_value = []
        
        tool = PubMedTool()
        tool.search_medical_cv_models(additional_terms="pathology slide analysis")
        
        call_args = mock_search.call_args
        query = call_args[0][0]
        
        # Should include base medical CV terms AND additional terms
        assert 'computer vision' in query.lower()
        assert 'deep learning' in query.lower()
        assert 'medical imaging' in query.lower()
        assert 'pathology slide analysis' in query
