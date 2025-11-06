"""
Comprehensive tests for agent protocol server.

Tests all endpoints, error handling, and server lifecycle.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from fastapi.testclient import TestClient

from research_viz_agent.agent_protocol.server import AgentServer, create_agent_server
from research_viz_agent.agent_protocol.schemas import (
    ResearchQuery,
    RAGSearchQuery,
    AgentRequest
)


class TestAgentServerInitialization:
    """Test AgentServer initialization."""
    
    def test_server_init_with_agent(self):
        """Test server initialization with provided agent."""
        mock_agent = MagicMock()
        server = AgentServer(
            agent=mock_agent,
            host="localhost",
            port=9999,
            agent_id="test-agent-123"
        )
        
        assert server.agent == mock_agent
        assert server.host == "localhost"
        assert server.port == 9999
        assert server.agent_id == "test-agent-123"
        assert server.app is not None
        assert isinstance(server._tasks, dict)
    
    @patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent')
    def test_server_init_without_agent(self, mock_agent_class):
        """Test server creates agent if not provided."""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        server = AgentServer(host="0.0.0.0", port=8000)
        
        mock_agent_class.assert_called_once()
        assert server.agent == mock_agent
        assert server.host == "0.0.0.0"
        assert server.port == 8000
        assert server.agent_id.startswith("research-viz-agent-")
    
    @patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent')
    def test_server_generates_agent_id(self, mock_agent_class):
        """Test server generates unique agent ID."""
        mock_agent_class.return_value = MagicMock()
        
        server1 = AgentServer()
        server2 = AgentServer()
        
        assert server1.agent_id != server2.agent_id
        assert server1.agent_id.startswith("research-viz-agent-")
        assert len(server1.agent_id) == 27  # "research-viz-agent-" + 8 hex chars


class TestAgentServerEndpoints:
    """Test server API endpoints."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.llm_provider = "github"
        agent.enable_rag = True
        agent.max_results = 20
        agent.llm = MagicMock()
        return agent
    
    @pytest.fixture
    def client(self, mock_agent):
        """Create test client."""
        server = AgentServer(agent=mock_agent, agent_id="test-server")
        return TestClient(server.app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns basic info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-server"
        assert data["name"] == "Medical CV Research Agent"
        assert data["version"] == "0.1.0"
        assert data["status"] == "online"
    
    def test_status_endpoint(self, client, mock_agent):
        """Test status endpoint returns capabilities."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-server"
        assert data["agent_name"] == "Medical CV Research Agent"
        assert data["version"] == "0.1.0"
        assert data["status"] == "online"
        assert len(data["capabilities"]) == 2
        
        # Check capabilities
        cap_names = [cap["name"] for cap in data["capabilities"]]
        assert "research_medical_cv" in cap_names
        assert "search_rag" in cap_names
        
        # Check metadata
        assert data["metadata"]["llm_provider"] == "github"
        assert data["metadata"]["rag_enabled"] is True
        assert data["metadata"]["max_results"] == 20
    
    def test_health_endpoint(self, client, mock_agent):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["agent_id"] == "test-server"
        assert data["rag_enabled"] is True
        assert data["llm_available"] is True
    
    def test_health_endpoint_no_llm(self, mock_agent):
        """Test health check when LLM is not available."""
        mock_agent.llm = None
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["llm_available"] is False


class TestResearchEndpoint:
    """Test /research endpoint."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with research results."""
        agent = MagicMock()
        agent.llm_provider = "github"
        agent.enable_rag = True
        agent.research.return_value = {
            'summary': 'Test summary of research',
            'total_papers': 15,
            'total_models': 3,
            'arxiv_results': [{'title': 'Paper 1'}],
            'pubmed_results': [{'title': 'Paper 2'}],
            'huggingface_results': [{'model_id': 'model1'}]
        }
        return agent
    
    @pytest.fixture
    def client(self, mock_agent):
        """Create test client."""
        server = AgentServer(agent=mock_agent)
        return TestClient(server.app)
    
    def test_research_endpoint_success(self, client, mock_agent):
        """Test successful research request."""
        query_data = {
            "query": "lung cancer detection",
            "max_results": 10,
            "enable_rag": True
        }
        
        response = client.post("/research", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "lung cancer detection"
        assert data["summary"] == "Test summary of research"
        assert data["total_papers"] == 15
        assert data["total_models"] == 3
        assert len(data["arxiv_results"]) == 1
        assert len(data["pubmed_results"]) == 1
        assert len(data["huggingface_results"]) == 1
        assert data["metadata"]["llm_provider"] == "github"
        
        # Verify agent.research was called
        mock_agent.research.assert_called_once_with("lung cancer detection")
    
    def test_research_endpoint_minimal_params(self, client, mock_agent):
        """Test research with minimal parameters."""
        query_data = {"query": "brain tumor"}
        
        response = client.post("/research", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "brain tumor"
        mock_agent.research.assert_called_once_with("brain tumor")
    
    def test_research_endpoint_empty_results(self, mock_agent):
        """Test research with no results."""
        mock_agent.research.return_value = {
            'summary': 'No results found',
            'total_papers': 0,
            'total_models': 0
        }
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/research", json={"query": "nonexistent topic"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_papers"] == 0
        assert data["total_models"] == 0
        assert data["arxiv_results"] == []
        assert data["pubmed_results"] == []
        assert data["huggingface_results"] == []
    
    def test_research_endpoint_agent_error(self, mock_agent):
        """Test research endpoint handles agent errors."""
        mock_agent.research.side_effect = Exception("Research failed")
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/research", json={"query": "test"})
        
        assert response.status_code == 500
        assert "Research failed" in response.json()["detail"]
    
    def test_research_endpoint_invalid_request(self, client):
        """Test research with invalid request data."""
        response = client.post("/research", json={"invalid": "data"})
        
        assert response.status_code == 422  # Validation error


class TestRAGSearchEndpoint:
    """Test /rag/search endpoint."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with RAG search results."""
        agent = MagicMock()
        agent.enable_rag = True
        agent.search_rag.return_value = {
            'total_count': 5,
            'results': [
                {'title': 'Doc 1', 'score': 0.95},
                {'title': 'Doc 2', 'score': 0.90},
                {'title': 'Doc 3', 'score': 0.85},
                {'title': 'Doc 4', 'score': 0.80},
                {'title': 'Doc 5', 'score': 0.75}
            ]
        }
        return agent
    
    @pytest.fixture
    def client(self, mock_agent):
        """Create test client."""
        server = AgentServer(agent=mock_agent)
        return TestClient(server.app)
    
    def test_rag_search_success(self, client, mock_agent):
        """Test successful RAG search."""
        query_data = {
            "query": "deep learning medical imaging",
            "k": 5
        }
        
        response = client.post("/rag/search", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "deep learning medical imaging"
        assert data["total_count"] == 5
        assert len(data["results"]) == 5
        assert data["source_filter"] is None
        
        mock_agent.search_rag.assert_called_once_with(
            query="deep learning medical imaging",
            k=5,
            source_filter=None
        )
    
    def test_rag_search_with_source_filter(self, client, mock_agent):
        """Test RAG search with source filter."""
        query_data = {
            "query": "CNN",
            "k": 10,
            "source_filter": "arxiv"
        }
        
        response = client.post("/rag/search", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["source_filter"] == "arxiv"
        
        mock_agent.search_rag.assert_called_once_with(
            query="CNN",
            k=10,
            source_filter="arxiv"
        )
    
    def test_rag_search_rag_disabled(self, mock_agent):
        """Test RAG search when RAG is disabled."""
        mock_agent.enable_rag = False
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/rag/search", json={"query": "test"})
        
        # HTTPException gets caught by outer exception handler and returns 500
        assert response.status_code == 500
        assert "RAG functionality is not enabled" in response.json()["detail"]
    
    def test_rag_search_empty_results(self, mock_agent):
        """Test RAG search with no results."""
        mock_agent.search_rag.return_value = {
            'total_count': 0,
            'results': []
        }
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/rag/search", json={"query": "nonexistent"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["results"] == []
    
    def test_rag_search_agent_error(self, mock_agent):
        """Test RAG search handles agent errors."""
        mock_agent.search_rag.side_effect = Exception("RAG search failed")
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/rag/search", json={"query": "test"})
        
        assert response.status_code == 500
        assert "RAG search failed" in response.json()["detail"]


class TestAgentRequestEndpoint:
    """Test generic /agent/request endpoint."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = MagicMock()
        agent.research.return_value = {
            'summary': 'Test',
            'total_papers': 10,
            'total_models': 2,
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': []
        }
        agent.search_rag.return_value = {
            'total_count': 3,
            'results': []
        }
        return agent
    
    @pytest.fixture
    def client(self, mock_agent):
        """Create test client."""
        server = AgentServer(agent=mock_agent)
        return TestClient(server.app)
    
    def test_agent_request_research_capability(self, client, mock_agent):
        """Test agent request for research capability."""
        request_data = {
            "request_id": "req-123",
            "capability": "research_medical_cv",
            "parameters": {
                "query": "lung cancer detection",
                "max_results": 20
            }
        }
        
        response = client.post("/agent/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-123"
        assert data["status"] == "success"
        assert "result" in data
        assert data["result"]["query"] == "lung cancer detection"
        assert data["metadata"]["capability"] == "research_medical_cv"
        
        mock_agent.research.assert_called_once_with("lung cancer detection")
    
    def test_agent_request_rag_search_capability(self, client, mock_agent):
        """Test agent request for RAG search capability."""
        request_data = {
            "request_id": "req-456",
            "capability": "search_rag",
            "parameters": {
                "query": "medical imaging",
                "k": 15
            }
        }
        
        response = client.post("/agent/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-456"
        assert data["status"] == "success"
        assert data["metadata"]["capability"] == "search_rag"
        
        mock_agent.search_rag.assert_called_once()
    
    def test_agent_request_unknown_capability(self, client):
        """Test agent request with unknown capability."""
        request_data = {
            "request_id": "req-789",
            "capability": "unknown_capability",
            "parameters": {}
        }
        
        response = client.post("/agent/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-789"
        assert data["status"] == "error"
        assert "Unknown capability" in data["error"]
        assert data["metadata"]["requested_capability"] == "unknown_capability"
    
    def test_agent_request_exception_handling(self, mock_agent):
        """Test agent request handles exceptions."""
        mock_agent.research.side_effect = ValueError("Invalid parameter")
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        request_data = {
            "request_id": "req-error",
            "capability": "research_medical_cv",
            "parameters": {"query": "test"}
        }
        
        response = client.post("/agent/request", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-error"
        assert data["status"] == "error"
        assert "Invalid parameter" in data["error"]
        assert data["metadata"]["exception_type"] == "ValueError"


class TestCreateAgentServer:
    """Test create_agent_server factory function."""
    
    @patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent')
    def test_create_agent_server_defaults(self, mock_agent_class):
        """Test factory creates server with defaults."""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        server = create_agent_server()
        
        assert isinstance(server, AgentServer)
        assert server.host == "0.0.0.0"
        assert server.port == 8000
        mock_agent_class.assert_called_once_with(llm_provider="github")
    
    @patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent')
    def test_create_agent_server_custom_params(self, mock_agent_class):
        """Test factory with custom parameters."""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        server = create_agent_server(
            llm_provider="openai",
            host="localhost",
            port=9000,
            model_name="gpt-4o-mini",
            max_results=50
        )
        
        assert server.host == "localhost"
        assert server.port == 9000
        mock_agent_class.assert_called_once_with(
            llm_provider="openai",
            model_name="gpt-4o-mini",
            max_results=50
        )
    
    @patch('research_viz_agent.agent_protocol.server.MedicalCVResearchAgent')
    def test_create_agent_server_with_rag_options(self, mock_agent_class):
        """Test factory with RAG options."""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        server = create_agent_server(
            llm_provider="github",
            enable_rag=False,
            rag_persist_dir="/custom/path"
        )
        
        mock_agent_class.assert_called_once_with(
            llm_provider="github",
            enable_rag=False,
            rag_persist_dir="/custom/path"
        )


class TestServerLifecycle:
    """Test server lifecycle methods."""
    
    @patch('research_viz_agent.agent_protocol.server.uvicorn.run')
    def test_server_run(self, mock_uvicorn_run):
        """Test server run method."""
        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent, host="localhost", port=9000)
        
        server.run()
        
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 9000
        assert call_kwargs["log_level"] == "info"
    
    @patch('research_viz_agent.agent_protocol.server.uvicorn.run')
    def test_server_run_with_custom_kwargs(self, mock_uvicorn_run):
        """Test server run with custom kwargs."""
        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent)
        
        server.run(workers=4, log_level="debug")
        
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["workers"] == 4
        assert call_kwargs["log_level"] == "debug"
    
    @pytest.mark.asyncio
    @patch('research_viz_agent.agent_protocol.server.uvicorn.Server')
    async def test_server_run_async(self, mock_server_class):
        """Test server async run method."""
        mock_server = MagicMock()
        mock_server.serve = MagicMock(return_value=None)
        mock_server_class.return_value = mock_server
        
        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent, host="localhost", port=9000)
        
        # Mock the serve method to be async
        async def mock_serve():
            pass
        mock_server.serve = mock_serve
        
        await server.run_async()
        
        mock_server_class.assert_called_once()


class TestServerCORS:
    """Test CORS middleware configuration."""
    
    def test_cors_enabled(self):
        """Test that CORS middleware is configured."""
        mock_agent = MagicMock()
        server = AgentServer(agent=mock_agent)
        
        # Check that CORS middleware is in the app
        middleware_classes = [m.cls.__name__ for m in server.app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestServerErrorHandling:
    """Test server error handling."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent that raises errors."""
        agent = MagicMock()
        agent.enable_rag = True
        return agent
    
    @pytest.fixture
    def client(self, mock_agent):
        """Create test client."""
        server = AgentServer(agent=mock_agent)
        return TestClient(server.app)
    
    def test_research_handles_timeout(self, mock_agent):
        """Test research endpoint handles timeout."""
        mock_agent.research.side_effect = TimeoutError("Request timeout")
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/research", json={"query": "test"})
        
        assert response.status_code == 500
        assert "Request timeout" in response.json()["detail"]
    
    def test_rag_search_handles_value_error(self, mock_agent):
        """Test RAG search handles ValueError."""
        mock_agent.search_rag.side_effect = ValueError("Invalid k value")
        server = AgentServer(agent=mock_agent)
        client = TestClient(server.app)
        
        response = client.post("/rag/search", json={"query": "test", "k": -1})
        
        assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
