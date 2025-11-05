"""
Unit tests for Medical CV Research Agent.
"""
import os
from unittest.mock import patch, MagicMock, Mock
from langchain_openai import ChatOpenAI

from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent


class TestMedicalCVAgentInitialization:
    """Tests for MedicalCVResearchAgent initialization."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_with_github_provider_default(self, mock_llm_factory, mock_workflow):
        """Test initialization with default GitHub provider."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        
        agent = MedicalCVResearchAgent(enable_rag=False)
        
        assert agent.llm_provider == "github"
        assert agent.skip_llm_init is False
        assert agent.llm == mock_llm
        mock_llm_factory.create_llm.assert_called_once()
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_init_with_openai_provider(self, mock_llm_factory, mock_workflow):
        """Test initialization with OpenAI provider."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        
        agent = MedicalCVResearchAgent(llm_provider="openai", enable_rag=False)
        
        assert agent.llm_provider == "openai"
        assert agent.skip_llm_init is False
        assert agent.llm == mock_llm
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_init_with_none_provider(self, mock_workflow):
        """Test initialization with 'none' provider."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        assert agent.llm_provider == "none"
        assert agent.skip_llm_init is True
        assert agent.llm is None
        assert agent.enable_rag is False
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_init_with_skip_llm_init_legacy(self, mock_workflow):
        """Test initialization with legacy skip_llm_init parameter."""
        agent = MedicalCVResearchAgent(skip_llm_init=True)
        
        assert agent.llm_provider == "none"
        assert agent.skip_llm_init is True
        assert agent.llm is None
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_with_custom_model_name(self, mock_llm_factory, mock_workflow):
        """Test initialization with custom model name."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        
        agent = MedicalCVResearchAgent(
            model_name="Llama-3.2-11B-Vision-Instruct",
            enable_rag=False
        )
        
        mock_llm_factory.create_llm.assert_called_once_with(
            provider="github",
            model_name="Llama-3.2-11B-Vision-Instruct",
            temperature=0.7,
            api_key="test-token"
        )
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_with_custom_temperature(self, mock_llm_factory, mock_workflow):
        """Test initialization with custom temperature."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        
        _ = MedicalCVResearchAgent(temperature=0.3, enable_rag=False)
        
        call_args = mock_llm_factory.create_llm.call_args
        assert call_args[1]['temperature'] == 0.3
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_falls_back_to_none(self, mock_llm_factory, mock_workflow):
        """Test initialization without API key falls back gracefully."""
        mock_llm_factory.create_llm.side_effect = ValueError("API key required")
        
        agent = MedicalCVResearchAgent(llm_provider="openai")
        
        assert agent.llm is None
        assert agent.skip_llm_init is True
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_arxiv_tool')
    @patch('research_viz_agent.agents.medical_cv_agent.create_pubmed_tool')
    @patch('research_viz_agent.agents.medical_cv_agent.create_huggingface_tool')
    def test_init_creates_mcp_tools(self, mock_hf, mock_pubmed, mock_arxiv, mock_workflow):
        """Test that MCP tools are created during initialization."""
        MedicalCVResearchAgent(llm_provider="none")
        
        mock_arxiv.assert_called_once()
        mock_pubmed.assert_called_once_with(email="research@example.com")
        mock_hf.assert_called_once()
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_huggingface_tool')
    @patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf-test-token"})
    def test_init_with_huggingface_token(self, mock_hf, mock_workflow):
        """Test initialization with HuggingFace token."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        mock_hf.assert_called_once_with(token="hf-test-token")
        assert agent.huggingface_token == "hf-test-token"
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_init_stores_max_results(self, mock_workflow):
        """Test that max_results is stored correctly."""
        agent = MedicalCVResearchAgent(llm_provider="none", max_results=50)
        
        assert agent.max_results == 50


class TestMedicalCVAgentRAGFunctionality:
    """Tests for RAG-related functionality."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_rag_store')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_with_rag_enabled_github(self, mock_llm_factory, mock_rag_store, mock_workflow):
        """Test RAG initialization with GitHub provider."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        mock_store = MagicMock()
        mock_rag_store.return_value = mock_store
        
        agent = MedicalCVResearchAgent(enable_rag=True)
        
        assert agent.enable_rag is True
        assert agent.rag_store == mock_store
        mock_rag_store.assert_called_once()
        # Check that it used GitHub embeddings
        call_args = mock_rag_store.call_args
        assert call_args[1]['embeddings_provider'] == 'github'
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_rag_store')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_init_with_rag_enabled_openai(self, mock_llm_factory, mock_rag_store, mock_workflow):
        """Test RAG initialization with OpenAI provider."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        mock_store = MagicMock()
        mock_rag_store.return_value = mock_store
        
        agent = MedicalCVResearchAgent(llm_provider="openai", enable_rag=True)
        
        assert agent.enable_rag is True
        assert agent.rag_store == mock_store
        # Check that it used OpenAI embeddings
        call_args = mock_rag_store.call_args
        assert call_args[1]['embeddings_provider'] == 'openai'
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_init_with_rag_disabled(self, mock_workflow):
        """Test that RAG is disabled when enable_rag=False."""
        agent = MedicalCVResearchAgent(llm_provider="none", enable_rag=False)
        
        assert agent.enable_rag is False
        assert agent.rag_store is None
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_init_rag_disabled_with_none_provider(self, mock_workflow):
        """Test that RAG is automatically disabled with 'none' provider."""
        agent = MedicalCVResearchAgent(llm_provider="none", enable_rag=True)
        
        assert agent.enable_rag is False
        assert agent.rag_store is None
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_rag_store')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_with_custom_rag_dir(self, mock_llm_factory, mock_rag_store, mock_workflow):
        """Test RAG initialization with custom persist directory."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        mock_store = MagicMock()
        mock_rag_store.return_value = mock_store
        
        agent = MedicalCVResearchAgent(
            enable_rag=True,
            rag_persist_dir="./custom_rag_dir"
        )
        
        call_args = mock_rag_store.call_args
        assert call_args[1]['persist_directory'] == "./custom_rag_dir"
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    @patch('research_viz_agent.agents.medical_cv_agent.create_rag_store')
    @patch('research_viz_agent.agents.medical_cv_agent.LLMFactory')
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
    def test_init_rag_failure_handled_gracefully(self, mock_llm_factory, mock_rag_store, mock_workflow):
        """Test that RAG initialization failure is handled gracefully."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_llm_factory.create_llm.return_value = mock_llm
        mock_rag_store.side_effect = Exception("RAG init failed")
        
        agent = MedicalCVResearchAgent(enable_rag=True)
        
        assert agent.enable_rag is False
        assert agent.rag_store is None


class TestMedicalCVAgentResearch:
    """Tests for research() method."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_research_basic(self, mock_workflow_class):
        """Test basic research functionality."""
        mock_workflow = MagicMock()
        mock_workflow_class.return_value = mock_workflow
        
        mock_workflow.run.return_value = {
            'query': 'lung cancer',
            'summary': 'Test summary',
            'arxiv_results': [{'title': 'Paper 1'}],
            'pubmed_results': [{'title': 'Paper 2'}],
            'huggingface_results': []
        }
        
        agent = MedicalCVResearchAgent(llm_provider="none")
        results = agent.research("lung cancer")
        
        assert results['query'] == 'lung cancer'
        assert results['summary'] == 'Test summary'
        assert results['total_papers'] == 2
        assert results['total_models'] == 0
        mock_workflow.run.assert_called_once_with("lung cancer")
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_research_with_rag_storage(self, mock_workflow_class):
        """Test research with RAG storage."""
        mock_workflow = MagicMock()
        mock_workflow_class.return_value = mock_workflow
        
        mock_workflow.run.return_value = {
            'query': 'test query',
            'summary': 'Summary',
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': []
        }
        
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        mock_rag_store = MagicMock()
        mock_rag_store.get_collection_info.return_value = {'document_count': 42}
        agent.rag_store = mock_rag_store
        
        results = agent.research("test query")
        
        assert results['query'] == 'test query'
        assert results['summary'] == 'Summary'
        assert results['total_papers'] == 0
        assert results['total_models'] == 0
        mock_rag_store.store_research_results.assert_called_once()
        mock_rag_store.get_collection_info.assert_called_once()
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_research_rag_storage_failure_handled(self, mock_workflow_class):
        """Test that RAG storage failure doesn't break research."""
        mock_workflow = MagicMock()
        mock_workflow_class.return_value = mock_workflow
        
        mock_workflow.run.return_value = {
            'query': 'test',
            'summary': 'Summary',
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': []
        }
        
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        mock_rag_store = MagicMock()
        mock_rag_store.store_research_results.side_effect = Exception("Storage failed")
        agent.rag_store = mock_rag_store
        
        # Should not raise exception
        results = agent.research("test")
        assert results['query'] == 'test'


class TestMedicalCVAgentFormatResults:
    """Tests for format_results() method."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_format_results_basic(self, mock_workflow):
        """Test basic result formatting."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        results = {
            'query': 'test query',
            'summary': 'This is a summary',
            'total_papers': 3,
            'total_models': 1,
            'arxiv_results': [
                {'title': 'ArXiv Paper', 'pdf_url': 'http://arxiv.org/1', 'published': '2024-01-01'}
            ],
            'pubmed_results': [
                {'title': 'PubMed Paper', 'pmid': '12345', 'journal': 'Nature'}
            ],
            'huggingface_results': [
                {'model_id': 'model/123', 'model_card_url': 'http://hf.co/model', 'downloads': 1000}
            ]
        }
        
        formatted = agent.format_results(results)
        
        assert 'RESEARCH SUMMARY: test query' in formatted
        assert 'Total Papers Found: 3' in formatted
        assert 'Total Models Found: 1' in formatted
        assert 'This is a summary' in formatted
        assert 'ArXiv Paper' in formatted
        assert 'PubMed Paper' in formatted
        assert 'model/123' in formatted
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_format_results_with_display_limit(self, mock_workflow):
        """Test result formatting with display limit."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        results = {
            'query': 'test',
            'summary': 'Summary',
            'total_papers': 10,
            'total_models': 0,
            'arxiv_results': [
                {'title': f'Paper {i}', 'pdf_url': f'url{i}', 'published': '2024'}
                for i in range(10)
            ],
            'pubmed_results': [],
            'huggingface_results': []
        }
        
        formatted = agent.format_results(results, display_limit=2)
        
        assert 'Paper 0' in formatted
        assert 'Paper 1' in formatted
        assert 'Paper 2' not in formatted
        assert '... and 8 more ArXiv papers' in formatted


class TestMedicalCVAgentRAGSearch:
    """Tests for RAG search functionality."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_search_rag_without_rag_enabled(self, mock_workflow):
        """Test RAG search when RAG is not enabled."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = False
        
        results = agent.search_rag("query")
        
        assert 'error' in results
        assert results['total_count'] == 0
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_search_rag_basic(self, mock_workflow):
        """Test basic RAG search."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {
            'source': 'arxiv',
            'type': 'paper',
            'title': 'Test Paper',
            'url': 'http://test.com'
        }
        
        mock_rag_store = MagicMock()
        mock_rag_store.similarity_search.return_value = [mock_doc]
        agent.rag_store = mock_rag_store
        
        results = agent.search_rag("deep learning", k=10)
        
        assert results['query'] == 'deep learning'
        assert results['total_count'] == 1
        assert results['results'][0]['title'] == 'Test Paper'
        assert results['results'][0]['source'] == 'arxiv'
        mock_rag_store.similarity_search.assert_called_once_with("deep learning", 10)
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_search_rag_with_source_filter(self, mock_workflow):
        """Test RAG search with source filter."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        
        mock_rag_store = MagicMock()
        mock_rag_store.search_by_source.return_value = []
        agent.rag_store = mock_rag_store
        
        results = agent.search_rag("query", k=5, source_filter="pubmed")
        
        assert results['source_filter'] == 'pubmed'
        mock_rag_store.search_by_source.assert_called_once_with("query", "pubmed", 5)
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_search_rag_error_handling(self, mock_workflow):
        """Test RAG search error handling."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        
        mock_rag_store = MagicMock()
        mock_rag_store.similarity_search.side_effect = Exception("Search failed")
        agent.rag_store = mock_rag_store
        
        results = agent.search_rag("query")
        
        assert 'error' in results
        assert 'Search failed' in results['error']


class TestMedicalCVAgentRAGStats:
    """Tests for RAG statistics functionality."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_get_rag_stats_without_rag(self, mock_workflow):
        """Test getting RAG stats when RAG is not enabled."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = False
        
        stats = agent.get_rag_stats()
        
        assert 'error' in stats
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_get_rag_stats_success(self, mock_workflow):
        """Test getting RAG stats successfully."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        
        mock_rag_store = MagicMock()
        mock_rag_store.get_collection_info.return_value = {
            'collection_name': 'test_collection',
            'document_count': 100,
            'persist_directory': './chroma_db'
        }
        agent.rag_store = mock_rag_store
        
        stats = agent.get_rag_stats()
        
        assert stats['collection_name'] == 'test_collection'
        assert stats['document_count'] == 100
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_get_rag_stats_error_handling(self, mock_workflow):
        """Test RAG stats error handling."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        agent.enable_rag = True
        
        mock_rag_store = MagicMock()
        mock_rag_store.get_collection_info.side_effect = Exception("Stats failed")
        agent.rag_store = mock_rag_store
        
        stats = agent.get_rag_stats()
        
        assert 'error' in stats


class TestMedicalCVAgentFormatRAGResults:
    """Tests for format_rag_results() method."""
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_format_rag_results_error(self, mock_workflow):
        """Test formatting RAG results with error."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        results = {'error': 'Something went wrong'}
        formatted = agent.format_rag_results(results)
        
        assert 'Error: Something went wrong' in formatted
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_format_rag_results_basic(self, mock_workflow):
        """Test basic RAG results formatting."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        results = {
            'query': 'deep learning',
            'total_count': 2,
            'source_filter': None,
            'results': [
                {
                    'title': 'Paper 1',
                    'source': 'arxiv',
                    'type': 'paper',
                    'url': 'http://arxiv.org/1',
                    'content': 'Full content here'
                },
                {
                    'title': 'Model 1',
                    'source': 'huggingface',
                    'type': 'model',
                    'url': 'http://hf.co/model',
                    'content': 'Model description'
                }
            ]
        }
        
        formatted = agent.format_rag_results(results)
        
        assert 'RAG SEARCH RESULTS: deep learning' in formatted
        assert 'Found 2 relevant documents' in formatted
        assert 'Paper 1' in formatted
        assert 'ARXIV' in formatted
        assert 'Model 1' in formatted
    
    @patch('research_viz_agent.agents.medical_cv_agent.ResearchWorkflow')
    def test_format_rag_results_with_content(self, mock_workflow):
        """Test RAG results formatting with content preview."""
        agent = MedicalCVResearchAgent(llm_provider="none")
        
        long_content = "A" * 300
        results = {
            'query': 'test',
            'total_count': 1,
            'source_filter': 'arxiv',
            'results': [
                {
                    'title': 'Test',
                    'source': 'arxiv',
                    'type': 'paper',
                    'url': 'http://test.com',
                    'content': long_content
                }
            ]
        }
        
        formatted = agent.format_rag_results(results, show_content=True)
        
        assert 'Preview:' in formatted
        assert '...' in formatted  # Content should be truncated
        assert 'Source Filter: arxiv' in formatted
