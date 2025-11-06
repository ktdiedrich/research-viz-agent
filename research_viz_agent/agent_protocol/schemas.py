"""
Agent Protocol Schemas

Defines the message formats and data structures for agent-to-agent communication.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class AgentCapability(BaseModel):
    """Describes a capability that an agent can perform."""
    
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Description of what this capability does")
    input_schema: Dict[str, Any] = Field(description="JSON schema for input parameters")
    output_schema: Dict[str, Any] = Field(description="JSON schema for output format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "research_medical_cv",
                "description": "Search and summarize medical computer vision research",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "total_papers": {"type": "integer"}
                    }
                }
            }
        }


class AgentStatus(BaseModel):
    """Status information about an agent."""
    
    agent_id: str = Field(description="Unique identifier for this agent")
    agent_name: str = Field(description="Human-readable name of the agent")
    version: str = Field(description="Agent version")
    status: Literal["online", "offline", "busy", "error"] = Field(
        default="online",
        description="Current operational status"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="List of capabilities this agent provides"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the agent"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "research-viz-agent-001",
                "agent_name": "Medical CV Research Agent",
                "version": "0.1.0",
                "status": "online",
                "capabilities": [],
                "metadata": {
                    "llm_provider": "github",
                    "rag_enabled": True
                }
            }
        }


class ResearchQuery(BaseModel):
    """Request to perform research on a topic."""
    
    query: str = Field(description="Research query string")
    max_results: int = Field(default=20, description="Maximum results per source")
    sources: Optional[List[Literal["arxiv", "pubmed", "huggingface"]]] = Field(
        default=None,
        description="Specific sources to search (None = all sources)"
    )
    enable_rag: bool = Field(
        default=True,
        description="Whether to store results in RAG database"
    )
    rag_search_only: bool = Field(
        default=False,
        description="Only search existing RAG database"
    )
    export_format: Optional[Literal["json", "csv", "text"]] = Field(
        default="json",
        description="Format for results export"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "lung cancer detection deep learning",
                "max_results": 20,
                "sources": ["arxiv", "pubmed"],
                "enable_rag": True,
                "rag_search_only": False,
                "export_format": "json"
            }
        }


class ResearchResult(BaseModel):
    """Results from a research query."""
    
    query: str = Field(description="Original query")
    summary: str = Field(description="AI-generated summary of findings")
    total_papers: int = Field(description="Total number of papers found")
    total_models: int = Field(description="Total number of models found")
    arxiv_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from ArXiv"
    )
    pubmed_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from PubMed"
    )
    huggingface_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from HuggingFace"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the search"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "lung cancer detection",
                "summary": "Recent advances in deep learning...",
                "total_papers": 45,
                "total_models": 12,
                "arxiv_results": [],
                "pubmed_results": [],
                "huggingface_results": [],
                "metadata": {
                    "timestamp": "2025-11-05T12:00:00Z",
                    "llm_provider": "github"
                }
            }
        }


class AgentRequest(BaseModel):
    """Standard request format for agent communication."""
    
    request_id: str = Field(description="Unique identifier for this request")
    capability: str = Field(description="Name of the capability to invoke")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the capability"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata (auth, tracking, etc.)"
    )
    timeout: Optional[int] = Field(
        default=300,
        description="Request timeout in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req-12345",
                "capability": "research_medical_cv",
                "parameters": {
                    "query": "brain tumor segmentation",
                    "max_results": 30
                },
                "metadata": {
                    "requesting_agent": "orchestrator-agent",
                    "priority": "normal"
                },
                "timeout": 300
            }
        }


class AgentResponse(BaseModel):
    """Standard response format for agent communication."""
    
    request_id: str = Field(description="ID of the request this responds to")
    status: Literal["success", "error", "partial", "timeout"] = Field(
        description="Status of the request"
    )
    result: Optional[Any] = Field(
        default=None,
        description="Result data (format depends on capability)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata (timing, resources used, etc.)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp (ISO 8601)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req-12345",
                "status": "success",
                "result": {
                    "query": "brain tumor segmentation",
                    "summary": "Analysis of research...",
                    "total_papers": 38
                },
                "error": None,
                "metadata": {
                    "execution_time_ms": 4523,
                    "sources_searched": ["arxiv", "pubmed", "huggingface"]
                },
                "timestamp": "2025-11-05T12:00:00.000Z"
            }
        }


class RAGSearchQuery(BaseModel):
    """Query for searching the RAG database."""
    
    query: str = Field(description="Search query")
    k: int = Field(default=10, description="Number of results to return")
    source_filter: Optional[Literal["arxiv", "pubmed", "huggingface"]] = Field(
        default=None,
        description="Filter by specific source"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "convolutional neural networks medical imaging",
                "k": 15,
                "source_filter": "arxiv"
            }
        }


class RAGSearchResult(BaseModel):
    """Results from RAG database search."""
    
    query: str = Field(description="Original query")
    total_count: int = Field(description="Number of results found")
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Search results with metadata"
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Source filter applied"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "CNN medical imaging",
                "total_count": 15,
                "results": [],
                "source_filter": None
            }
        }
