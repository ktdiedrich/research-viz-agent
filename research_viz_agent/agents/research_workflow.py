"""
LangGraph workflow for medical computer vision research agent.
"""
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import operator


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
    
    def __init__(self, llm: ChatOpenAI, arxiv_tool, pubmed_tool, huggingface_tool):
        """
        Initialize the research workflow.
        
        Args:
            llm: Language model for generating summaries
            arxiv_tool: Tool for searching arXiv
            pubmed_tool: Tool for searching PubMed
            huggingface_tool: Tool for searching HuggingFace
        """
        self.llm = llm
        self.arxiv_tool = arxiv_tool
        self.pubmed_tool = pubmed_tool
        self.huggingface_tool = huggingface_tool
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
    
    def _search_arxiv(self, state: AgentState) -> AgentState:
        """Search arXiv for relevant papers."""
        query = state.get("query", "")
        print(f"Searching arXiv for: {query}")
        
        results = self.arxiv_tool.search_medical_cv_models(query)
        state["arxiv_results"] = results
        return state
    
    def _search_pubmed(self, state: AgentState) -> AgentState:
        """Search PubMed for relevant papers."""
        query = state.get("query", "")
        print(f"Searching PubMed for: {query}")
        
        results = self.pubmed_tool.search_medical_cv_models(query)
        state["pubmed_results"] = results
        return state
    
    def _search_huggingface(self, state: AgentState) -> AgentState:
        """Search HuggingFace for relevant models."""
        query = state.get("query", "")
        print(f"Searching HuggingFace for: {query}")
        
        results = self.huggingface_tool.search_medical_cv_models(query)
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
        
        # Generate summary
        messages = prompt.format_messages()
        response = self.llm.invoke(messages)
        
        state["summary"] = response.content
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
