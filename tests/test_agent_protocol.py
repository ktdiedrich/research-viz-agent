"""
Tests for agent-to-agent communication protocol.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from research_viz_agent.agent_protocol.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    AgentCapability,
    ResearchQuery,
    ResearchResult,
    RAGSearchQuery,
    RAGSearchResult
)
from research_viz_agent.agent_protocol.server import AgentServer
from research_viz_agent.agent_protocol.client import AgentClient


class TestSchemas:
    """Test Pydantic schema models."""
    
    def test_agent_capability_creation(self):
        """Test creating an AgentCapability."""
        capability = AgentCapability(
            name="test_capability",
            description="Test description",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        assert capability.name == "test_capability"
        assert capability.description == "Test description"
    
    def test_agent_status_creation(self):
        """Test creating an AgentStatus."""
        status = AgentStatus(
            agent_id="test-001",
            agent_name="Test Agent",
            version="0.1.0",
            status="online",
            capabilities=[]
        )
        assert status.agent_id == "test-001"
        assert status.status == "online"
        assert len(status.capabilities) == 0
    
    def test_research_query_validation(self):
        """Test ResearchQuery validation."""
        query = ResearchQuery(
            query="test query",
            max_results=10,
            sources=["arxiv", "pubmed"]
        )
        assert query.query == "test query"
        assert query.max_results == 10
        assert "arxiv" in query.sources
    
    def test_research_query_defaults(self):
        """Test ResearchQuery default values."""
        query = ResearchQuery(query="test")
        assert query.max_results == 20
        assert query.sources is None
        assert query.enable_rag is True
        assert query.rag_search_only is False
    
    def test_research_result_creation(self):
        """Test creating a ResearchResult."""
        result = ResearchResult(
            query="test query",
            summary="test summary",
            total_papers=10,
            total_models=5,
            arxiv_results=[{"title": "Test Paper"}],
            metadata={"timestamp": "2025-01-01"}
        )
        assert result.query == "test query"
        assert result.total_papers == 10
        assert len(result.arxiv_results) == 1
    
    def test_agent_request_creation(self):
        """Test creating an AgentRequest."""
        request = AgentRequest(
            request_id="req-123",
            capability="research_medical_cv",
            parameters={"query": "test"},
            timeout=300
        )
        assert request.request_id == "req-123"
        assert request.capability == "research_medical_cv"
        assert request.timeout == 300
    
    def test_agent_response_creation(self):
        """Test creating an AgentResponse."""
        response = AgentResponse(
            request_id="req-123",
            status="success",
            result={"data": "test"}
        )
        assert response.request_id == "req-123"
        assert response.status == "success"
        assert response.result["data"] == "test"
        assert response.timestamp is not None
    
    def test_agent_response_error(self):
        """Test creating an error AgentResponse."""
        response = AgentResponse(
            request_id="req-123",
            status="error",
            error="Something went wrong"
        )
        assert response.status == "error"
        assert response.error == "Something went wrong"
        assert response.result is None
    
    def test_rag_search_query_creation(self):
        """Test creating a RAGSearchQuery."""
        query = RAGSearchQuery(
            query="test query",
            k=5,
            source_filter="arxiv"
        )
        assert query.query == "test query"
        assert query.k == 5
        assert query.source_filter == "arxiv"
    
    def test_rag_search_result_creation(self):
        """Test creating a RAGSearchResult."""
        result = RAGSearchResult(
            query="test query",
            total_count=3,
            results=[
                {"title": "Paper 1"},
                {"title": "Paper 2"},
                {"title": "Paper 3"}
            ]
        )
        assert result.query == "test query"
        assert result.total_count == 3
        assert len(result.results) == 3


class TestAgentServer:
    """Test AgentServer functionality."""
    
    def test_server_initialization(self):
        """Test creating an AgentServer."""
        mock_agent = Mock()
        server = AgentServer(
            agent=mock_agent,
            host="localhost",
            port=9000,
            agent_id="test-server"
        )
        assert server.host == "localhost"
        assert server.port == 9000
        assert server.agent_id == "test-server"
        assert server.agent == mock_agent
    
    def test_server_default_initialization(self):
        """Test server with default parameters."""
        with patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent'):
            server = AgentServer()
            assert server.host == "0.0.0.0"
            assert server.port == 8000
            assert server.agent_id.startswith("research-viz-agent-")
    
    def test_server_creates_app(self):
        """Test that server creates a FastAPI app."""
        mock_agent = Mock()
        server = AgentServer(agent=mock_agent)
        assert server.app is not None
        assert hasattr(server.app, "routes")


class TestAgentClient:
    """Test AgentClient functionality."""
    
    def test_client_initialization(self):
        """Test creating an AgentClient."""
        client = AgentClient(
            base_url="http://localhost:9000",
            timeout=60
        )
        assert client.base_url == "http://localhost:9000"
        assert client.timeout == 60
    
    def test_client_strips_trailing_slash(self):
        """Test that base_url trailing slash is removed."""
        client = AgentClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"
    
    def test_client_context_manager(self):
        """Test client as context manager."""
        with AgentClient(base_url="http://localhost:8000") as client:
            assert client is not None
        # Client should be closed after context
    
    @pytest.mark.asyncio
    async def test_client_async_context_manager(self):
        """Test client as async context manager."""
        async with AgentClient(base_url="http://localhost:8000") as client:
            assert client is not None
        # Client should be closed after context


class TestSchemaValidation:
    """Test schema validation and serialization."""
    
    def test_research_query_json_serialization(self):
        """Test ResearchQuery can be serialized to JSON."""
        query = ResearchQuery(
            query="test query",
            max_results=10,
            sources=["arxiv"]
        )
        json_data = query.model_dump()
        assert json_data["query"] == "test query"
        assert json_data["max_results"] == 10
        assert json_data["sources"] == ["arxiv"]
    
    def test_agent_response_json_deserialization(self):
        """Test AgentResponse can be created from JSON."""
        json_data = {
            "request_id": "req-123",
            "status": "success",
            "result": {"data": "test"},
            "error": None,
            "metadata": {},
            "timestamp": "2025-01-01T00:00:00Z"
        }
        response = AgentResponse(**json_data)
        assert response.request_id == "req-123"
        assert response.status == "success"
    
    def test_invalid_status_rejected(self):
        """Test that invalid status values are rejected."""
        with pytest.raises(ValueError):
            AgentStatus(
                agent_id="test",
                agent_name="Test",
                version="1.0",
                status="invalid_status"  # Should fail
            )
    
    def test_invalid_source_filter_rejected(self):
        """Test that invalid source filters are rejected."""
        with pytest.raises(ValueError):
            RAGSearchQuery(
                query="test",
                source_filter="invalid_source"  # Should fail
            )


class TestEndToEndFlow:
    """Test complete request/response flows."""
    
    def test_research_request_response_flow(self):
        """Test complete research request/response cycle."""
        # Create request
        request = AgentRequest(
            request_id="req-001",
            capability="research_medical_cv",
            parameters={
                "query": "lung cancer detection",
                "max_results": 20
            }
        )
        
        # Simulate response
        response = AgentResponse(
            request_id=request.request_id,
            status="success",
            result={
                "query": "lung cancer detection",
                "summary": "Test summary",
                "total_papers": 15,
                "total_models": 3
            }
        )
        
        assert response.request_id == request.request_id
        assert response.status == "success"
        assert response.result["total_papers"] == 15
    
    def test_rag_search_flow(self):
        """Test RAG search request/response cycle."""
        # Create query
        query = RAGSearchQuery(
            query="deep learning",
            k=10,
            source_filter="arxiv"
        )
        
        # Simulate result
        result = RAGSearchResult(
            query=query.query,
            total_count=8,
            results=[{"title": f"Paper {i}"} for i in range(8)],
            source_filter=query.source_filter
        )
        
        assert result.query == query.query
        assert result.total_count == 8
        assert len(result.results) == 8
        assert result.source_filter == "arxiv"
    
    def test_error_response_flow(self):
        """Test error handling in request/response cycle."""
        request = AgentRequest(
            request_id="req-error",
            capability="unknown_capability",
            parameters={}
        )
        
        response = AgentResponse(
            request_id=request.request_id,
            status="error",
            error="Unknown capability: unknown_capability"
        )
        
        assert response.status == "error"
        assert "Unknown capability" in response.error
        assert response.result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
