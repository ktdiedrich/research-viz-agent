"""
Main medical computer vision research agent.
"""
import os
from typing import Optional, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from research_viz_agent.mcp_tools.arxiv_tool import create_arxiv_tool
from research_viz_agent.mcp_tools.pubmed_tool import create_pubmed_tool
from research_viz_agent.mcp_tools.huggingface_tool import create_huggingface_tool
from research_viz_agent.agents.research_workflow import ResearchWorkflow
from research_viz_agent.utils.rag_store import create_rag_store
from research_viz_agent.utils.llm_factory import LLMFactory, LLMProvider


class MedicalCVResearchAgent:
    """
    AI Agent for summarizing capabilities and uses of medical computer vision AI models
    from scientific research and model collections.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider = "openai",
        openai_api_key: Optional[str] = None,
        github_token: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        pubmed_email: str = "research@example.com",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_results: int = 20,
        enable_rag: bool = True,
        rag_persist_dir: str = "./chroma_db",
        skip_llm_init: bool = False
    ):
        """
        Initialize the Medical CV Research Agent.
        
        Args:
            llm_provider: LLM provider ("openai", "github", or "none")
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            github_token: GitHub token (or set GITHUB_TOKEN env var) 
            huggingface_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
            pubmed_email: Email for PubMed API
            model_name: Model name to use (provider-specific defaults if not specified)
            temperature: Temperature for LLM responses
            max_results: Maximum number of results to fetch from each source
            enable_rag: Whether to enable RAG storage of results
            rag_persist_dir: Directory to persist ChromaDB data
            skip_llm_init: Legacy parameter for backward compatibility (sets llm_provider to "none")
        """
        # Load environment variables
        load_dotenv()
        
        # Handle legacy skip_llm_init parameter
        if skip_llm_init:
            llm_provider = "none"
        
        self.llm_provider = llm_provider
        self.skip_llm_init = (llm_provider == "none")
        
        # Initialize LLM based on provider
        if llm_provider == "none":
            self.llm = None
            self.api_key = None
            print("⚠ Skipping LLM initialization - AI summarization disabled")
        else:
            try:
                # Set up API key based on provider
                if llm_provider == "openai":
                    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
                    self.api_key = api_key
                elif llm_provider == "github":
                    api_key = github_token or os.getenv("GITHUB_TOKEN")
                    self.api_key = api_key
                else:
                    raise ValueError(f"Unsupported LLM provider: {llm_provider}")
                
                # Create LLM instance
                self.llm = LLMFactory.create_llm(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key
                )
                
                print(f"✓ LLM initialized: {llm_provider} ({model_name or 'default model'})")
                
            except Exception as e:
                print(f"⚠ LLM initialization failed: {e}")
                print("  Continuing without AI summarization...")
                self.llm = None
                self.api_key = None
                self.skip_llm_init = True
        
        self.huggingface_token = huggingface_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize MCP tools
        self.arxiv_tool = create_arxiv_tool()
        self.pubmed_tool = create_pubmed_tool(email=pubmed_email)
        self.huggingface_tool = create_huggingface_tool(token=self.huggingface_token)
        
        # Store max_results for workflow
        self.max_results = max_results
        
        # Initialize RAG store if enabled (requires OpenAI for embeddings)
        self.enable_rag = enable_rag and not self.skip_llm_init and llm_provider in ["openai", "github"]
        self.rag_store = None
        
        if self.enable_rag:
            try:
                # For RAG embeddings, we need an OpenAI-compatible API
                # Use OpenAI key for embeddings even with GitHub provider
                embeddings_api_key = None
                if llm_provider == "openai":
                    embeddings_api_key = self.api_key
                elif llm_provider == "github":
                    # For GitHub provider, try to use OpenAI key for embeddings if available
                    embeddings_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
                    if not embeddings_api_key:
                        print("⚠ OpenAI API key required for RAG embeddings with GitHub provider")
                        self.enable_rag = False
                
                if self.enable_rag:
                    self.rag_store = create_rag_store(
                        persist_directory=rag_persist_dir,
                        openai_api_key=embeddings_api_key
                    )
                    print(f"✓ RAG store initialized at {rag_persist_dir}")
            except Exception as e:
                print(f"⚠ RAG store initialization failed: {e}")
                print("  Continuing without RAG functionality...")
                self.enable_rag = False
        elif self.skip_llm_init:
            print("⚠ RAG functionality disabled (requires LLM provider for embeddings)")
        elif llm_provider not in ["openai", "github"]:
            print(f"⚠ RAG functionality not supported with {llm_provider} provider")
        
        # Initialize workflow
        self.workflow = ResearchWorkflow(
            llm=self.llm,
            arxiv_tool=self.arxiv_tool,
            pubmed_tool=self.pubmed_tool,
            huggingface_tool=self.huggingface_tool,
            max_results=max_results
        )
    
    def research(self, query: str) -> Dict:
        """
        Research medical computer vision AI models based on a query.
        
        Args:
            query: The research query (e.g., "lung cancer detection", "skin lesion classification")
            
        Returns:
            Dictionary containing:
                - summary: Comprehensive summary of findings
                - arxiv_results: List of arXiv papers
                - pubmed_results: List of PubMed papers
                - huggingface_results: List of HuggingFace models
        """
        print(f"\n{'='*60}")
        print(f"Starting research on: {query}")
        print(f"{'='*60}\n")
        
        # Run the workflow
        results = self.workflow.run(query)
        
        # Store results in RAG store if enabled
        if self.enable_rag and self.rag_store:
            try:
                print("📚 Storing results in RAG database...")
                self.rag_store.store_research_results(
                    arxiv_results=results.get('arxiv_results', []),
                    pubmed_results=results.get('pubmed_results', []),
                    huggingface_results=results.get('huggingface_results', []),
                    query=query
                )
                
                # Get collection info
                info = self.rag_store.get_collection_info()
                print(f"✓ Results stored. Total documents in RAG: {info.get('document_count', 'unknown')}")
            except Exception as e:
                print(f"⚠ Failed to store results in RAG: {e}")
        
        return {
            'query': results.get('query', ''),
            'summary': results.get('summary', ''),
            'arxiv_results': results.get('arxiv_results', []),
            'pubmed_results': results.get('pubmed_results', []),
            'huggingface_results': results.get('huggingface_results', []),
            'total_papers': len(results.get('arxiv_results', [])) + len(results.get('pubmed_results', [])),
            'total_models': len(results.get('huggingface_results', []))
        }
    
    def format_results(self, results: Dict, display_limit: int = 5) -> str:
        """
        Format research results for display.
        
        Args:
            results: Results dictionary from research()
            display_limit: Number of results to display from each source
            
        Returns:
            Formatted string of results
        """
        output = []
        
        output.append(f"\n{'='*80}")
        output.append(f"RESEARCH SUMMARY: {results['query']}")
        output.append(f"{'='*80}\n")
        
        output.append(f"Total Papers Found: {results['total_papers']}")
        output.append(f"Total Models Found: {results['total_models']}\n")
        
        output.append(f"\n{'-'*80}")
        output.append("AI-GENERATED SUMMARY")
        output.append(f"{'-'*80}\n")
        output.append(results['summary'])
        
        output.append(f"\n\n{'-'*80}")
        output.append(f"DETAILED SOURCES")
        output.append(f"{'-'*80}\n")
        
        # ArXiv papers
        if results['arxiv_results']:
            output.append("\n### ArXiv Papers ###\n")
            for i, paper in enumerate(results['arxiv_results'][:display_limit], 1):
                output.append(f"{i}. {paper.get('title', 'N/A')}")
                output.append(f"   URL: {paper.get('pdf_url', 'N/A')}")
                output.append(f"   Published: {paper.get('published', 'N/A')}\n")
            
            if len(results['arxiv_results']) > display_limit:
                output.append(f"   ... and {len(results['arxiv_results']) - display_limit} more ArXiv papers\n")
        
        # PubMed papers
        if results['pubmed_results']:
            output.append("\n### PubMed Papers ###\n")
            for i, paper in enumerate(results['pubmed_results'][:display_limit], 1):
                output.append(f"{i}. {paper.get('title', 'N/A')}")
                output.append(f"   PMID: {paper.get('pmid', 'N/A')}")
                output.append(f"   Journal: {paper.get('journal', 'N/A')}\n")
            
            if len(results['pubmed_results']) > display_limit:
                output.append(f"   ... and {len(results['pubmed_results']) - display_limit} more PubMed papers\n")
        
        # HuggingFace models
        if results['huggingface_results']:
            output.append("\n### HuggingFace Models ###\n")
            for i, model in enumerate(results['huggingface_results'][:display_limit], 1):
                output.append(f"{i}. {model.get('model_id', 'N/A')}")
                output.append(f"   URL: {model.get('model_card_url', 'N/A')}")
                output.append(f"   Downloads: {model.get('downloads', 0)}\n")
            
            if len(results['huggingface_results']) > display_limit:
                output.append(f"   ... and {len(results['huggingface_results']) - display_limit} more HuggingFace models\n")
        
        return "\n".join(output)
    
    def search_rag(self, query: str, k: int = 5, source_filter: Optional[str] = None) -> Dict:
        """
        Search the RAG database for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            source_filter: Filter by source ('arxiv', 'pubmed', 'huggingface')
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.enable_rag or not self.rag_store:
            return {
                'error': 'RAG functionality not available',
                'results': [],
                'total_count': 0
            }
        
        try:
            if source_filter:
                documents = self.rag_store.search_by_source(query, source_filter, k)
            else:
                documents = self.rag_store.similarity_search(query, k)
            
            results = []
            for doc in documents:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': doc.metadata.get('type', 'unknown'),
                    'title': doc.metadata.get('title', 'N/A'),
                    'url': doc.metadata.get('url', 'N/A')
                })
            
            return {
                'query': query,
                'results': results,
                'total_count': len(results),
                'source_filter': source_filter
            }
            
        except Exception as e:
            return {
                'error': f'RAG search failed: {str(e)}',
                'results': [],
                'total_count': 0
            }
    
    def get_rag_stats(self) -> Dict:
        """
        Get statistics about the RAG database.
        
        Returns:
            Dictionary with RAG database statistics
        """
        if not self.enable_rag or not self.rag_store:
            return {'error': 'RAG functionality not available'}
        
        try:
            return self.rag_store.get_collection_info()
        except Exception as e:
            return {'error': f'Failed to get RAG stats: {str(e)}'}
    
    def format_rag_results(self, rag_results: Dict, show_content: bool = False) -> str:
        """
        Format RAG search results for display.
        
        Args:
            rag_results: Results from search_rag()
            show_content: Whether to show document content
            
        Returns:
            Formatted string of RAG results
        """
        if 'error' in rag_results:
            return f"Error: {rag_results['error']}"
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"RAG SEARCH RESULTS: {rag_results['query']}")
        output.append(f"{'='*60}\n")
        
        if rag_results['source_filter']:
            output.append(f"Source Filter: {rag_results['source_filter']}")
        
        output.append(f"Found {rag_results['total_count']} relevant documents\n")
        
        for i, result in enumerate(rag_results['results'], 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   Source: {result['source'].upper()}")
            output.append(f"   Type: {result['type'].title()}")
            output.append(f"   URL: {result['url']}")
            
            if show_content:
                # Show first 200 characters of content
                content_preview = result['content'][:200].replace('\n', ' ').strip()
                if len(result['content']) > 200:
                    content_preview += "..."
                output.append(f"   Preview: {content_preview}")
            
            output.append("")  # Empty line between results
        
        return "\n".join(output)
