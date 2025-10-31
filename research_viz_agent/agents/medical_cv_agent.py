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


class MedicalCVResearchAgent:
    """
    AI Agent for summarizing capabilities and uses of medical computer vision AI models
    from scientific research and model collections.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        pubmed_email: str = "research@example.com",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_results: int = 20
    ):
        """
        Initialize the Medical CV Research Agent.
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            huggingface_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
            pubmed_email: Email for PubMed API
            model_name: OpenAI model to use
            temperature: Temperature for LLM responses
            max_results: Maximum number of results to fetch from each source
        """
        # Load environment variables
        load_dotenv()
        
        # Set up API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.huggingface_token = huggingface_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize MCP tools
        self.arxiv_tool = create_arxiv_tool()
        self.pubmed_tool = create_pubmed_tool(email=pubmed_email)
        self.huggingface_tool = create_huggingface_tool(token=self.huggingface_token)
        
        # Store max_results for workflow
        self.max_results = max_results
        
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
