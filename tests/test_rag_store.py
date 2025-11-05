"""
Unit tests for the ResearchRAGStore.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from langchain_core.documents import Document

from research_viz_agent.utils.rag_store import (
    ResearchRAGStore,
    create_rag_store
)


class TestResearchRAGStoreInitialization:
    """Test ResearchRAGStore initialization."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_init_with_defaults(self, mock_create_embeddings, mock_chroma):
        """Test initialization with default parameters."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        assert store.persist_directory == "./chroma_db"
        assert store.collection_name == "medical_cv_research"
        assert store.embeddings_provider == "openai"
        mock_create_embeddings.assert_called_once_with(
            provider="openai",
            api_key=None,
            base_url=None
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_init_with_custom_params(self, mock_create_embeddings, mock_chroma):
        """Test initialization with custom parameters."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore(
            persist_directory="/custom/path",
            collection_name="custom_collection",
            embeddings_provider="github",
            api_key="test_key",
            base_url="http://test.com"
        )
        
        assert store.persist_directory == "/custom/path"
        assert store.collection_name == "custom_collection_github"
        assert store.embeddings_provider == "github"
        mock_create_embeddings.assert_called_once_with(
            provider="github",
            api_key="test_key",
            base_url="http://test.com"
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_init_with_legacy_openai_key(self, mock_create_embeddings, mock_chroma):
        """Test initialization with legacy openai_api_key parameter."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore(
            openai_api_key="legacy_key"
        )
        
        mock_create_embeddings.assert_called_once_with(
            provider="openai",
            api_key="legacy_key",
            base_url=None
        )
    
    @patch('research_viz_agent.utils.rag_store.os.makedirs')
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_init_handles_settings_conflict(self, mock_create_embeddings, mock_chroma, mock_makedirs):
        """Test initialization handles ChromaDB settings conflict."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        # First call raises ValueError, second succeeds
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        
        mock_chroma.side_effect = [
            ValueError("different settings conflict"),
            mock_vector_store
        ]
        
        store = ResearchRAGStore(
            persist_directory="./chroma_db",
            embeddings_provider="github"
        )
        
        # Should create provider-specific directory
        assert store.persist_directory == "./chroma_db_github"
        mock_makedirs.assert_called_once_with("./chroma_db_github", exist_ok=True)
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_init_provider_suffix_added(self, mock_create_embeddings, mock_chroma):
        """Test that provider suffix is added to collection name for non-openai."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore(
            collection_name="test",
            embeddings_provider="github"
        )
        
        assert store.collection_name == "test_github"


class TestResearchRAGStoreCreateDocumentId:
    """Test document ID creation."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_create_document_id_basic(self, mock_create_embeddings, mock_chroma):
        """Test basic document ID creation."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        doc_id = store._create_document_id("arxiv", "12345", "Test Paper")
        
        assert isinstance(doc_id, str)
        assert len(doc_id) == 32  # MD5 hash length
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_create_document_id_missing_identifier(self, mock_create_embeddings, mock_chroma):
        """Test document ID creation with missing identifier."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        doc_id = store._create_document_id("pubmed", "", "Fallback Title")
        
        assert isinstance(doc_id, str)
        assert len(doc_id) == 32


class TestResearchRAGStoreAddArxivResults:
    """Test adding ArXiv results."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_arxiv_results_basic(self, mock_create_embeddings, mock_chroma):
        """Test adding ArXiv results to store."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        results = [
            {
                'title': 'Test Paper',
                'authors': ['Author A', 'Author B'],
                'summary': 'This is a test abstract',
                'categories': ['cs.CV', 'cs.AI'],
                'primary_category': 'cs.CV',
                'published': '2024-01-01',
                'updated': '2024-01-02',
                'pdf_url': 'http://arxiv.org/pdf/1234.5678',
                'entry_id': 'http://arxiv.org/abs/1234.5678'
            }
        ]
        
        store.add_arxiv_results(results, "test query")
        
        mock_vector_store.add_documents.assert_called_once()
        docs = mock_vector_store.add_documents.call_args[0][0]
        assert len(docs) == 1
        assert 'Test Paper' in docs[0].page_content
        assert docs[0].metadata['source'] == 'arxiv'
        assert docs[0].metadata['query'] == 'test query'
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_arxiv_results_empty_list(self, mock_create_embeddings, mock_chroma):
        """Test adding empty ArXiv results list."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.add_arxiv_results([], "query")
        
        mock_vector_store.add_documents.assert_not_called()
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_arxiv_results_handles_duplicates(self, mock_create_embeddings, mock_chroma):
        """Test handling duplicate errors when adding ArXiv results."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.add_documents.side_effect = [
            Exception("unique constraint violation"),
            None  # Second call succeeds
        ]
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        results = [{'title': 'Paper', 'authors': [], 'entry_id': '123'}]
        
        # Should handle duplicate gracefully
        store.add_arxiv_results(results, "query")
        
        # Should be called twice (once fails, then per-document)
        assert mock_vector_store.add_documents.call_count >= 1


class TestResearchRAGStoreAddPubMedResults:
    """Test adding PubMed results."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_pubmed_results_basic(self, mock_create_embeddings, mock_chroma):
        """Test adding PubMed results to store."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        results = [
            {
                'title': 'Medical Study',
                'authors': ['Researcher X', 'Researcher Y'],
                'abstract': 'Study abstract',
                'journal': 'Medical Journal',
                'pmid': '12345',
                'mesh_terms': ['Term1', 'Term2'],
                'keywords': ['keyword1'],
                'publication_date': '2024 Jan'
            }
        ]
        
        store.add_pubmed_results(results, "medical query")
        
        mock_vector_store.add_documents.assert_called_once()
        docs = mock_vector_store.add_documents.call_args[0][0]
        assert len(docs) == 1
        assert 'Medical Study' in docs[0].page_content
        assert docs[0].metadata['source'] == 'pubmed'
        assert docs[0].metadata['pmid'] == '12345'
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_pubmed_results_empty_list(self, mock_create_embeddings, mock_chroma):
        """Test adding empty PubMed results list."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.add_pubmed_results([], "query")
        
        mock_vector_store.add_documents.assert_not_called()


class TestResearchRAGStoreAddHuggingFaceResults:
    """Test adding HuggingFace results."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_huggingface_results_basic(self, mock_create_embeddings, mock_chroma):
        """Test adding HuggingFace results to store."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        results = [
            {
                'model_id': 'test/model-1',
                'author': 'test_org',
                'pipeline_tag': 'image-classification',
                'tags': ['medical', 'cv'],
                'library_name': 'transformers',
                'downloads': 1000,
                'likes': 50,
                'created_at': '2024-01-01',
                'last_modified': '2024-01-02',
                'model_card_url': 'https://huggingface.co/test/model-1'
            }
        ]
        
        store.add_huggingface_results(results, "model query")
        
        mock_vector_store.add_documents.assert_called_once()
        docs = mock_vector_store.add_documents.call_args[0][0]
        assert len(docs) == 1
        assert 'test/model-1' in docs[0].page_content
        assert docs[0].metadata['source'] == 'huggingface'
        assert docs[0].metadata['type'] == 'model'
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_add_huggingface_results_empty_list(self, mock_create_embeddings, mock_chroma):
        """Test adding empty HuggingFace results list."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.add_huggingface_results([], "query")
        
        mock_vector_store.add_documents.assert_not_called()


class TestResearchRAGStoreStoreResults:
    """Test store_research_results method."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_store_research_results_all_sources(self, mock_create_embeddings, mock_chroma):
        """Test storing results from all sources."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        arxiv = [{'title': 'ArXiv', 'authors': [], 'entry_id': '1'}]
        pubmed = [{'title': 'PubMed', 'authors': [], 'pmid': '2'}]
        hf = [{'model_id': 'model', 'author': 'org'}]
        
        store.store_research_results(
            arxiv_results=arxiv,
            pubmed_results=pubmed,
            huggingface_results=hf,
            query="test"
        )
        
        # Should be called 3 times (once per source)
        assert mock_vector_store.add_documents.call_count == 3
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_store_research_results_partial(self, mock_create_embeddings, mock_chroma):
        """Test storing results from some sources."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        arxiv = [{'title': 'ArXiv', 'authors': [], 'entry_id': '1'}]
        
        store.store_research_results(
            arxiv_results=arxiv,
            query="test"
        )
        
        # Should only be called once
        assert mock_vector_store.add_documents.call_count == 1


class TestResearchRAGStoreSearch:
    """Test search methods."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_similarity_search_basic(self, mock_create_embeddings, mock_chroma):
        """Test basic similarity search."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_doc = Document(page_content="Test", metadata={"source": "arxiv"})
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search.return_value = [mock_doc]
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        results = store.similarity_search("test query", k=5)
        
        assert len(results) == 1
        assert results[0].page_content == "Test"
        mock_vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=5,
            filter=None
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_similarity_search_with_filter(self, mock_create_embeddings, mock_chroma):
        """Test similarity search with filter."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search.return_value = []
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.similarity_search("query", k=3, filter_dict={"source": "arxiv"})
        
        mock_vector_store.similarity_search.assert_called_once_with(
            query="query",
            k=3,
            filter={"source": "arxiv"}
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_similarity_search_with_score(self, mock_create_embeddings, mock_chroma):
        """Test similarity search with scores."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_doc = Document(page_content="Test", metadata={})
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        results = store.similarity_search_with_score("query")
        
        assert len(results) == 1
        assert results[0][0].page_content == "Test"
        assert results[0][1] == 0.95
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_search_by_source(self, mock_create_embeddings, mock_chroma):
        """Test search by source."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search.return_value = []
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.search_by_source("query", "pubmed", k=10)
        
        mock_vector_store.similarity_search.assert_called_once_with(
            query="query",
            k=10,
            filter={"source": "pubmed"}
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_search_by_type(self, mock_create_embeddings, mock_chroma):
        """Test search by type."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search.return_value = []
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.search_by_type("query", "model", k=7)
        
        mock_vector_store.similarity_search.assert_called_once_with(
            query="query",
            k=7,
            filter={"type": "model"}
        )


class TestResearchRAGStoreRetriever:
    """Test retriever methods."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_get_retriever_default(self, mock_create_embeddings, mock_chroma):
        """Test getting retriever with default parameters."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_retriever = MagicMock()
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        retriever = store.get_retriever()
        
        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5}
        )
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_get_retriever_custom_kwargs(self, mock_create_embeddings, mock_chroma):
        """Test getting retriever with custom search kwargs."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        store.get_retriever(search_kwargs={"k": 10, "filter": {"source": "arxiv"}})
        
        mock_vector_store.as_retriever.assert_called_once_with(
            search_kwargs={"k": 10, "filter": {"source": "arxiv"}}
        )


class TestResearchRAGStoreCollectionInfo:
    """Test collection info methods."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_get_collection_info_success(self, mock_create_embeddings, mock_chroma):
        """Test getting collection info successfully."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        
        mock_vector_store = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        info = store.get_collection_info()
        
        assert info["collection_name"] == "medical_cv_research"
        assert info["document_count"] == 42
        assert info["persist_directory"] == "./chroma_db"
        assert "error" not in info
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_get_collection_info_error(self, mock_create_embeddings, mock_chroma):
        """Test getting collection info with error."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        mock_vector_store = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        info = store.get_collection_info()
        
        assert info["document_count"] == 0
        assert "error" in info
        assert "Collection not found" in info["error"]


class TestCreateRAGStoreFactory:
    """Test factory function."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_create_rag_store_factory(self, mock_create_embeddings, mock_chroma):
        """Test factory function creates ResearchRAGStore."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = create_rag_store(
            persist_directory="/test/path",
            collection_name="test_collection",
            embeddings_provider="github",
            api_key="test_key"
        )
        
        assert isinstance(store, ResearchRAGStore)
        assert store.persist_directory == "/test/path"
        assert store.embeddings_provider == "github"
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_create_rag_store_with_legacy_param(self, mock_create_embeddings, mock_chroma):
        """Test factory function with legacy openai_api_key."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = create_rag_store(
            openai_api_key="legacy_key"
        )
        
        assert isinstance(store, ResearchRAGStore)


class TestResearchRAGStoreIntegration:
    """Test integration scenarios."""
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_full_workflow_store_and_search(self, mock_create_embeddings, mock_chroma):
        """Test full workflow of storing and searching."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_doc = Document(
            page_content="Medical AI paper",
            metadata={"source": "arxiv"}
        )
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_vector_store.similarity_search.return_value = [mock_doc]
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        # Store results
        arxiv_results = [
            {
                'title': 'Medical AI',
                'authors': ['Researcher'],
                'summary': 'Abstract',
                'entry_id': '1234',
                'categories': ['cs.CV']
            }
        ]
        store.add_arxiv_results(arxiv_results, "medical AI")
        
        # Search
        results = store.similarity_search("medical AI", k=5)
        
        assert len(results) == 1
        assert "Medical AI paper" in results[0].page_content
    
    @patch('research_viz_agent.utils.rag_store.Chroma')
    @patch('research_viz_agent.utils.rag_store.LLMFactory.create_embeddings')
    def test_handles_missing_fields_gracefully(self, mock_create_embeddings, mock_chroma):
        """Test handling of missing fields in results."""
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        mock_client = MagicMock()
        mock_vector_store._client = mock_client
        mock_chroma.return_value = mock_vector_store
        
        store = ResearchRAGStore()
        
        # Results with minimal/missing fields
        results = [
            {
                'title': 'Minimal Paper',
                # Missing most fields
            }
        ]
        
        # Should not raise error
        store.add_arxiv_results(results, "query")
        
        mock_vector_store.add_documents.assert_called_once()
        docs = mock_vector_store.add_documents.call_args[0][0]
        assert len(docs) == 1
