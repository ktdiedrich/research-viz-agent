"""
Agent Client

Client for making requests to other agents using the standardized protocol.
"""
import uuid
import httpx
from typing import Optional, Dict, Any, List
from research_viz_agent.agent_protocol.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    ResearchQuery,
    ResearchResult,
    RAGSearchQuery,
    RAGSearchResult
)


class AgentClient:
    """
    Client for communicating with other agents via HTTP.
    
    Provides high-level methods for invoking agent capabilities
    and handles request/response formatting.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the agent client.
        
        Args:
            base_url: Base URL of the agent server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            headers: Optional HTTP headers (for auth, etc.)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}
        self.client = httpx.Client(timeout=timeout, headers=self.headers)
        self.async_client = httpx.AsyncClient(timeout=timeout, headers=self.headers)
    
    def get_status(self) -> AgentStatus:
        """
        Get the status and capabilities of the agent.
        
        Returns:
            AgentStatus object with agent information
        """
        response = self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return AgentStatus(**response.json())
    
    async def get_status_async(self) -> AgentStatus:
        """Async version of get_status()."""
        response = await self.async_client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return AgentStatus(**response.json())
    
    def research(
        self,
        query: str,
        max_results: int = 20,
        sources: Optional[List[str]] = None,
        enable_rag: bool = True
    ) -> ResearchResult:
        """
        Request the agent to perform research on a topic.
        
        Args:
            query: Research query
            max_results: Maximum results per source
            sources: Specific sources to search
            enable_rag: Whether to store in RAG database
        
        Returns:
            ResearchResult with findings and summary
        """
        request_data = ResearchQuery(
            query=query,
            max_results=max_results,
            sources=sources,
            enable_rag=enable_rag
        )
        
        response = self.client.post(
            f"{self.base_url}/research",
            json=request_data.model_dump()
        )
        response.raise_for_status()
        return ResearchResult(**response.json())
    
    async def research_async(
        self,
        query: str,
        max_results: int = 20,
        sources: Optional[List[str]] = None,
        enable_rag: bool = True
    ) -> ResearchResult:
        """Async version of research()."""
        request_data = ResearchQuery(
            query=query,
            max_results=max_results,
            sources=sources,
            enable_rag=enable_rag
        )
        
        response = await self.async_client.post(
            f"{self.base_url}/research",
            json=request_data.model_dump()
        )
        response.raise_for_status()
        return ResearchResult(**response.json())
    
    def search_rag(
        self,
        query: str,
        k: int = 10,
        source_filter: Optional[str] = None
    ) -> RAGSearchResult:
        """
        Search the agent's RAG database.
        
        Args:
            query: Search query
            k: Number of results to return
            source_filter: Filter by source (arxiv, pubmed, huggingface)
        
        Returns:
            RAGSearchResult with relevant documents
        """
        request_data = RAGSearchQuery(
            query=query,
            k=k,
            source_filter=source_filter
        )
        
        response = self.client.post(
            f"{self.base_url}/rag/search",
            json=request_data.model_dump()
        )
        response.raise_for_status()
        return RAGSearchResult(**response.json())
    
    async def search_rag_async(
        self,
        query: str,
        k: int = 10,
        source_filter: Optional[str] = None
    ) -> RAGSearchResult:
        """Async version of search_rag()."""
        request_data = RAGSearchQuery(
            query=query,
            k=k,
            source_filter=source_filter
        )
        
        response = await self.async_client.post(
            f"{self.base_url}/rag/search",
            json=request_data.model_dump()
        )
        response.raise_for_status()
        return RAGSearchResult(**response.json())
    
    def send_request(
        self,
        capability: str,
        parameters: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Send a generic agent request using the standard protocol.
        
        Args:
            capability: Name of the capability to invoke
            parameters: Parameters for the capability
            request_id: Optional request ID (generated if not provided)
        
        Returns:
            AgentResponse with result or error
        """
        request = AgentRequest(
            request_id=request_id or f"req-{uuid.uuid4().hex[:12]}",
            capability=capability,
            parameters=parameters,
            timeout=self.timeout
        )
        
        response = self.client.post(
            f"{self.base_url}/agent/request",
            json=request.model_dump()
        )
        response.raise_for_status()
        return AgentResponse(**response.json())
    
    async def send_request_async(
        self,
        capability: str,
        parameters: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> AgentResponse:
        """Async version of send_request()."""
        request = AgentRequest(
            request_id=request_id or f"req-{uuid.uuid4().hex[:12]}",
            capability=capability,
            parameters=parameters,
            timeout=self.timeout
        )
        
        response = await self.async_client.post(
            f"{self.base_url}/agent/request",
            json=request.model_dump()
        )
        response.raise_for_status()
        return AgentResponse(**response.json())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the agent is healthy and responding.
        
        Returns:
            Health status information
        """
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async version of health_check()."""
        response = await self.async_client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP client connections."""
        self.client.close()
    
    async def close_async(self):
        """Close the async HTTP client connections."""
        await self.async_client.aclose()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.close_async()
