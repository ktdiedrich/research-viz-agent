"""
LangGraph workflow for medical computer vision research agent.
"""
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import backoff
from openai import RateLimitError


class AgentState(TypedDict):
    """State definition for the research agent."""
    query: str
    arxiv_results: Annotated[List[Dict], operator.add]
    pubmed_results: Annotated[List[Dict], operator.add]
    huggingface_results: Annotated[List[Dict], operator.add]
    summary: str
    next_step: str


class ResearchWorkflow:
    """LangGraph workflow for coordinating research across multiple sources."""
    
    def __init__(self, llm: Optional[ChatOpenAI], arxiv_tool, pubmed_tool, huggingface_tool, max_results: int = 20):
        """
        Initialize the research workflow.
        
        Args:
            llm: Language model for generating summaries
            arxiv_tool: Tool for searching arXiv
            pubmed_tool: Tool for searching PubMed
            huggingface_tool: Tool for searching HuggingFace
            max_results: Maximum number of results to fetch from each source
        """
        self.llm = llm
        self.arxiv_tool = arxiv_tool
        self.pubmed_tool = pubmed_tool
        self.huggingface_tool = huggingface_tool
        self.max_results = max_results
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search_arxiv", self._search_arxiv)
        workflow.add_node("search_pubmed", self._search_pubmed)
        workflow.add_node("search_huggingface", self._search_huggingface)
        workflow.add_node("summarize", self._summarize_results)
        
        # Define the flow
        workflow.set_entry_point("search_arxiv")
        workflow.add_edge("search_arxiv", "search_pubmed")
        workflow.add_edge("search_pubmed", "search_huggingface")
        workflow.add_edge("search_huggingface", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError,),  # Only retry on rate limits, not quota errors
        max_tries=3,
        factor=2,
        max_value=60
    )
    def _invoke_llm_with_backoff(self, messages):
        """Invoke LLM with exponential backoff for rate limiting."""
        if self.llm is None:
            raise ValueError("LLM not available - summarization disabled")
        return self.llm.invoke(messages)
    
    def _search_arxiv(self, state: AgentState) -> AgentState:
        """Search arXiv for relevant papers."""
        query = state.get("query", "")
        print(f"Searching arXiv for: {query}")
        
        results = self.arxiv_tool.search_medical_cv_models(query, max_results=self.max_results)
        state["arxiv_results"] = results
        return state
    
    def _search_pubmed(self, state: AgentState) -> AgentState:
        """Search PubMed for relevant papers."""
        query = state.get("query", "")
        print(f"Searching PubMed for: {query}")
        
        results = self.pubmed_tool.search_medical_cv_models(query, max_results=self.max_results)
        state["pubmed_results"] = results
        return state
    
    def _search_huggingface(self, state: AgentState) -> AgentState:
        """Search HuggingFace for relevant models."""
        query = state.get("query", "")
        print(f"Searching HuggingFace for: {query}")
        
        results = self.huggingface_tool.search_medical_cv_models(query, max_results=self.max_results)
        state["huggingface_results"] = results
        return state
    
    def _summarize_results(self, state: AgentState) -> AgentState:
        """Summarize all research findings."""
        print("Generating summary...")
        
        query = state.get("query", "")
        arxiv_results = state.get("arxiv_results", [])
        pubmed_results = state.get("pubmed_results", [])
        huggingface_results = state.get("huggingface_results", [])
        
        # Prepare context for summarization
        context = self._prepare_context(arxiv_results, pubmed_results, huggingface_results)
        
        # Create prompt for summarization
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert AI researcher specializing in medical computer vision. "
                "Your task is to analyze and summarize the capabilities and uses of AI models "
                "for medical computer vision based on research papers and available models."
            )),
            HumanMessage(content=(
                f"User Query: {query}\n\n"
                f"Research Findings:\n{context}\n\n"
                "Please provide a comprehensive summary that includes:\n"
                "1. Overview of key AI models and their capabilities\n"
                "2. Common medical imaging applications and use cases\n"
                "3. Notable research findings and trends\n"
                "4. Available pre-trained models and their purposes\n"
                "5. Recommendations for practitioners\n\n"
                "Make the summary informative, well-structured, and actionable."
            ))
        ])
        
        # Generate summary with rate limiting
        messages = prompt.format_messages()
        try:
            if self.llm is None:
                # Skip AI summarization, create fallback summary
                fallback_summary = self._create_fallback_summary(arxiv_results, pubmed_results, huggingface_results, query)
                state["summary"] = f"""
🤖 AI Summarization Disabled

{fallback_summary}

Note: AI-powered summarization was skipped. To enable AI summarization, run without --no-summary flag and ensure your OpenAI API key is configured.
"""
            else:
                response = self._invoke_llm_with_backoff(messages)
                state["summary"] = response.content
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating summary: {e}")
            
            # Provide specific guidance based on error type
            if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                fallback_summary = self._create_fallback_summary(arxiv_results, pubmed_results, huggingface_results, query)
                state["summary"] = f"""
⚠️ OpenAI API Quota Exceeded

Your OpenAI API quota has been exceeded. Please check your billing and usage at: https://platform.openai.com/account/billing

Here's a basic summary of the research findings:

{fallback_summary}

To resolve this issue:
1. Check your OpenAI account billing: https://platform.openai.com/account/billing
2. Upgrade your plan or add credits if needed
3. Wait for your quota to reset if on a free tier
4. Use --no-rag flag to skip AI summarization and just get raw results

You can still search the RAG database if it contains previous results:
  python -m research_viz_agent.cli --rag-search "{query}"
"""
            elif "rate_limit" in error_msg.lower():
                state["summary"] = f"Rate limit exceeded. Please wait a moment and try again. The research results were still collected successfully."
            else:
                state["summary"] = f"Summary generation failed: {error_msg}\n\nThe research results were still collected successfully. You can view them below."
        
        return state
    
    def _prepare_context(
        self,
        arxiv_results: List[Dict],
        pubmed_results: List[Dict],
        huggingface_results: List[Dict]
    ) -> str:
        """Prepare research context for summarization."""
        context_parts = []
        
        # ArXiv papers
        if arxiv_results:
            context_parts.append("=== ArXiv Papers ===")
            for i, paper in enumerate(arxiv_results[:10], 1):
                context_parts.append(
                    f"\n{i}. {paper.get('title', 'N/A')}\n"
                    f"   Authors: {', '.join(paper.get('authors', [])[:3])}\n"
                    f"   Summary: {paper.get('summary', 'N/A')[:300]}...\n"
                    f"   Categories: {', '.join(paper.get('categories', []))}\n"
                )
        
        # PubMed papers
        if pubmed_results:
            context_parts.append("\n=== PubMed Papers ===")
            for i, paper in enumerate(pubmed_results[:10], 1):
                context_parts.append(
                    f"\n{i}. {paper.get('title', 'N/A')}\n"
                    f"   Authors: {', '.join(paper.get('authors', [])[:3])}\n"
                    f"   Abstract: {paper.get('abstract', 'N/A')[:300]}...\n"
                    f"   Journal: {paper.get('journal', 'N/A')}\n"
                )
        
        # HuggingFace models
        if huggingface_results:
            context_parts.append("\n=== HuggingFace Models ===")
            for i, model in enumerate(huggingface_results[:10], 1):
                context_parts.append(
                    f"\n{i}. {model.get('model_id', 'N/A')}\n"
                    f"   Author: {model.get('author', 'N/A')}\n"
                    f"   Task: {model.get('pipeline_tag', 'N/A')}\n"
                    f"   Downloads: {model.get('downloads', 0)}\n"
                    f"   Tags: {', '.join(model.get('tags', [])[:5])}\n"
                )
        
        return "\n".join(context_parts)
    
    def _create_fallback_summary(
        self,
        arxiv_results: List[Dict],
        pubmed_results: List[Dict], 
        huggingface_results: List[Dict],
        query: str
    ) -> str:
        """Create a basic summary without LLM when API is unavailable."""
        summary_parts = []
        
        total_papers = len(arxiv_results) + len(pubmed_results)
        total_models = len(huggingface_results)
        
        summary_parts.append(f"Research Query: {query}")
        summary_parts.append(f"Total Papers Found: {total_papers} ({len(arxiv_results)} ArXiv, {len(pubmed_results)} PubMed)")
        summary_parts.append(f"Total Models Found: {total_models} (HuggingFace)")
        summary_parts.append("")
        
        if arxiv_results:
            summary_parts.append("ArXiv Research Highlights:")
            for i, paper in enumerate(arxiv_results[:3], 1):
                title = paper.get('title', 'N/A')[:100] + ("..." if len(paper.get('title', '')) > 100 else "")
                summary_parts.append(f"  {i}. {title}")
            if len(arxiv_results) > 3:
                summary_parts.append(f"  ... and {len(arxiv_results) - 3} more papers")
            summary_parts.append("")
        
        if pubmed_results:
            summary_parts.append("PubMed Research Highlights:")
            for i, paper in enumerate(pubmed_results[:3], 1):
                title = paper.get('title', 'N/A')[:100] + ("..." if len(paper.get('title', '')) > 100 else "")
                summary_parts.append(f"  {i}. {title}")
            if len(pubmed_results) > 3:
                summary_parts.append(f"  ... and {len(pubmed_results) - 3} more papers")
            summary_parts.append("")
        
        if huggingface_results:
            summary_parts.append("HuggingFace Model Highlights:")
            for i, model in enumerate(huggingface_results[:3], 1):
                model_id = model.get('model_id', 'N/A')
                task = model.get('pipeline_tag', 'unknown')
                downloads = model.get('downloads', 0)
                summary_parts.append(f"  {i}. {model_id} ({task}) - {downloads:,} downloads")
            if len(huggingface_results) > 3:
                summary_parts.append(f"  ... and {len(huggingface_results) - 3} more models")
        
        return "\n".join(summary_parts)
    
    def run(self, query: str) -> Dict:
        """
        Run the research workflow.
        
        Args:
            query: User's research query
            
        Returns:
            Final state with summary and results
        """
        initial_state = {
            "query": query,
            "arxiv_results": [],
            "pubmed_results": [],
            "huggingface_results": [],
            "summary": "",
            "next_step": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state
