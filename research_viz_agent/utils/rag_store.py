"""
RAG (Retrieval-Augmented Generation) store using ChromaDB for research results.
"""
import os
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ResearchRAGStore:
    """
    RAG store for indexing and retrieving research results from ArXiv, PubMed, and HuggingFace.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "medical_cv_research",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            openai_api_key: OpenAI API key for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize Langchain Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def _create_document_id(self, source: str, identifier: str) -> str:
        """Create a unique document ID."""
        return hashlib.md5(f"{source}:{identifier}".encode()).hexdigest()
    
    def add_arxiv_results(self, results: List[Dict], query: str) -> None:
        """
        Add ArXiv results to the RAG store.
        
        Args:
            results: List of ArXiv paper dictionaries
            query: Original search query
        """
        documents = []
        
        for paper in results:
            doc_id = self._create_document_id("arxiv", paper.get("entry_id", ""))
            
            # Create document content
            content = f"""
Title: {paper.get('title', 'N/A')}

Authors: {', '.join(paper.get('authors', []))}

Abstract: {paper.get('summary', 'N/A')}

Categories: {', '.join(paper.get('categories', []))}

Primary Category: {paper.get('primary_category', 'N/A')}

Published: {paper.get('published', 'N/A')}
Updated: {paper.get('updated', 'N/A')}
            """.strip()
            
            # Create metadata
            metadata = {
                "source": "arxiv",
                "type": "paper",
                "query": query,
                "title": paper.get('title', ''),
                "authors": ', '.join(paper.get('authors', [])[:3]),
                "url": paper.get('pdf_url', ''),
                "entry_id": paper.get('entry_id', ''),
                "categories": ', '.join(paper.get('categories', [])),
                "published": paper.get('published', ''),
                "indexed_at": datetime.now().isoformat()
            }
            
            documents.append(Document(
                page_content=content,
                metadata=metadata,
                id=doc_id
            ))
        
        if documents:
            self.vector_store.add_documents(documents)
    
    def add_pubmed_results(self, results: List[Dict], query: str) -> None:
        """
        Add PubMed results to the RAG store.
        
        Args:
            results: List of PubMed paper dictionaries
            query: Original search query
        """
        documents = []
        
        for paper in results:
            doc_id = self._create_document_id("pubmed", paper.get("pmid", ""))
            
            # Create document content
            content = f"""
Title: {paper.get('title', 'N/A')}

Authors: {', '.join(paper.get('authors', []))}

Abstract: {paper.get('abstract', 'N/A')}

Journal: {paper.get('journal', 'N/A')}

MeSH Terms: {', '.join(paper.get('mesh_terms', []))}

Keywords: {', '.join(paper.get('keywords', []))}

Published: {paper.get('publication_date', 'N/A')}
            """.strip()
            
            # Create metadata
            metadata = {
                "source": "pubmed",
                "type": "paper",
                "query": query,
                "title": paper.get('title', ''),
                "authors": ', '.join(paper.get('authors', [])[:3]),
                "pmid": paper.get('pmid', ''),
                "journal": paper.get('journal', ''),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid', '')}/",
                "mesh_terms": ', '.join(paper.get('mesh_terms', [])[:5]),
                "publication_date": paper.get('publication_date', ''),
                "indexed_at": datetime.now().isoformat()
            }
            
            documents.append(Document(
                page_content=content,
                metadata=metadata,
                id=doc_id
            ))
        
        if documents:
            self.vector_store.add_documents(documents)
    
    def add_huggingface_results(self, results: List[Dict], query: str) -> None:
        """
        Add HuggingFace model results to the RAG store.
        
        Args:
            results: List of HuggingFace model dictionaries
            query: Original search query
        """
        documents = []
        
        for model in results:
            doc_id = self._create_document_id("huggingface", model.get("model_id", ""))
            
            # Create document content
            content = f"""
Model ID: {model.get('model_id', 'N/A')}

Author: {model.get('author', 'N/A')}

Task: {model.get('pipeline_tag', 'N/A')}

Tags: {', '.join(model.get('tags', []))}

Library: {model.get('library_name', 'N/A')}

Downloads: {model.get('downloads', 0):,}

Likes: {model.get('likes', 0)}

Created: {model.get('created_at', 'N/A')}
Last Modified: {model.get('last_modified', 'N/A')}
            """.strip()
            
            # Create metadata
            metadata = {
                "source": "huggingface",
                "type": "model",
                "query": query,
                "model_id": model.get('model_id', ''),
                "author": model.get('author', ''),
                "task": model.get('pipeline_tag', ''),
                "url": model.get('model_card_url', ''),
                "tags": ', '.join(model.get('tags', [])[:10]),
                "downloads": model.get('downloads', 0),
                "likes": model.get('likes', 0),
                "library": model.get('library_name', ''),
                "indexed_at": datetime.now().isoformat()
            }
            
            documents.append(Document(
                page_content=content,
                metadata=metadata,
                id=doc_id
            ))
        
        if documents:
            self.vector_store.add_documents(documents)
    
    def store_research_results(
        self, 
        arxiv_results: List[Dict] = None,
        pubmed_results: List[Dict] = None,
        huggingface_results: List[Dict] = None,
        query: str = ""
    ) -> None:
        """
        Store all research results in the RAG store.
        
        Args:
            arxiv_results: ArXiv search results
            pubmed_results: PubMed search results
            huggingface_results: HuggingFace search results
            query: Original search query
        """
        if arxiv_results:
            self.add_arxiv_results(arxiv_results, query)
            
        if pubmed_results:
            self.add_pubmed_results(pubmed_results, query)
            
        if huggingface_results:
            self.add_huggingface_results(huggingface_results, query)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search in the RAG store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of tuples (document, relevance_score)
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a Langchain retriever for the vector store.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            Langchain retriever object
        """
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 5}
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def search_by_source(
        self,
        query: str,
        source: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search documents from a specific source.
        
        Args:
            query: Search query
            source: Source filter ('arxiv', 'pubmed', 'huggingface')
            k: Number of results to return
            
        Returns:
            List of documents from the specified source
        """
        return self.similarity_search(
            query=query,
            k=k,
            filter_dict={"source": source}
        )
    
    def search_by_type(
        self,
        query: str,
        doc_type: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search documents by type.
        
        Args:
            query: Search query
            doc_type: Document type ('paper' or 'model')
            k: Number of results to return
            
        Returns:
            List of documents of the specified type
        """
        return self.similarity_search(
            query=query,
            k=k,
            filter_dict={"type": doc_type}
        )


def create_rag_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "medical_cv_research",
    openai_api_key: Optional[str] = None
) -> ResearchRAGStore:
    """
    Factory function to create a ResearchRAGStore instance.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        openai_api_key: OpenAI API key for embeddings
        
    Returns:
        ResearchRAGStore instance
    """
    return ResearchRAGStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        openai_api_key=openai_api_key
    )