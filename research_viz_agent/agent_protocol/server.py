"""
Agent Server

HTTP server that exposes the research agent's capabilities via RESTful API,
enabling other agents to communicate with it.
"""
import asyncio
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
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


class AgentServer:
    """
    HTTP server for exposing research agent capabilities to other agents.
    
    Provides RESTful endpoints for:
    - Agent status and capabilities
    - Research queries
    - RAG database searches
    - Asynchronous task handling
    """
    
    def __init__(
        self,
        agent: Optional[MedicalCVResearchAgent] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the agent server.
        
        Args:
            agent: MedicalCVResearchAgent instance (created if not provided)
            host: Host to bind the server to
            port: Port to bind the server to
            agent_id: Unique identifier for this agent instance
        """
        self.agent = agent or MedicalCVResearchAgent()
        self.host = host
        self.port = port
        self.agent_id = agent_id or f"research-viz-agent-{uuid.uuid4().hex[:8]}"
        self.app = self._create_app()
        self._tasks: Dict[str, Any] = {}  # Track async tasks
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            print(f"🚀 Agent Server starting: {self.agent_id}")
            print(f"   Listening on http://{self.host}:{self.port}")
            yield
            # Shutdown
            print(f"🛑 Agent Server shutting down: {self.agent_id}")
        
        app = FastAPI(
            title="Medical CV Research Agent API",
            description="Agent-to-agent communication interface for medical computer vision research",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # Enable CORS for cross-origin requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define endpoints
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with basic info."""
            return {
                "agent_id": self.agent_id,
                "name": "Medical CV Research Agent",
                "version": "0.1.0",
                "status": "online"
            }
        
        @app.get("/status", response_model=AgentStatus)
        async def get_status():
            """Get agent status and capabilities."""
            return AgentStatus(
                agent_id=self.agent_id,
                agent_name="Medical CV Research Agent",
                version="0.1.0",
                status="online",
                capabilities=[
                    AgentCapability(
                        name="research_medical_cv",
                        description="Search and summarize medical computer vision research from ArXiv, PubMed, and HuggingFace",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "enable_rag": {"type": "boolean"}
                            },
                            "required": ["query"]
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "total_papers": {"type": "integer"},
                                "total_models": {"type": "integer"}
                            }
                        }
                    ),
                    AgentCapability(
                        name="search_rag",
                        description="Search the persistent RAG database for relevant medical CV research",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "k": {"type": "integer"},
                                "source_filter": {"type": "string"}
                            },
                            "required": ["query"]
                        },
                        output_schema={
                            "type": "object",
                            "properties": {
                                "total_count": {"type": "integer"},
                                "results": {"type": "array"}
                            }
                        }
                    )
                ],
                metadata={
                    "llm_provider": self.agent.llm_provider,
                    "rag_enabled": self.agent.enable_rag,
                    "max_results": self.agent.max_results
                }
            )
        
        @app.post("/research", response_model=ResearchResult)
        async def research(query: ResearchQuery):
            """
            Perform research on a medical CV topic.
            
            This endpoint executes the full research workflow:
            - Searches ArXiv, PubMed, and/or HuggingFace
            - Stores results in RAG database (if enabled)
            - Generates AI summary of findings
            """
            try:
                # Execute research
                results = self.agent.research(
                    query.query,
                    # Note: sources filtering would need to be added to agent
                )
                
                # Format response
                return ResearchResult(
                    query=query.query,
                    summary=results.get('summary', ''),
                    total_papers=results.get('total_papers', 0),
                    total_models=results.get('total_models', 0),
                    arxiv_results=results.get('arxiv_results', []),
                    pubmed_results=results.get('pubmed_results', []),
                    huggingface_results=results.get('huggingface_results', []),
                    metadata={
                        "llm_provider": self.agent.llm_provider,
                        "rag_enabled": self.agent.enable_rag
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/rag/search", response_model=RAGSearchResult)
        async def search_rag(query: RAGSearchQuery):
            """
            Search the RAG database for relevant documents.
            
            This searches the persistent vector database without making
            new API calls to external sources.
            """
            try:
                if not self.agent.enable_rag:
                    raise HTTPException(
                        status_code=400,
                        detail="RAG functionality is not enabled on this agent"
                    )
                
                results = self.agent.search_rag(
                    query=query.query,
                    k=query.k,
                    source_filter=query.source_filter
                )
                
                return RAGSearchResult(
                    query=query.query,
                    total_count=results.get('total_count', 0),
                    results=results.get('results', []),
                    source_filter=query.source_filter
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/agent/request", response_model=AgentResponse)
        async def handle_agent_request(request: AgentRequest):
            """
            Generic endpoint for agent-to-agent communication.
            
            Handles requests using the standardized AgentRequest/AgentResponse format.
            Routes to appropriate capability based on request.capability field.
            """
            try:
                if request.capability == "research_medical_cv":
                    # Convert parameters to ResearchQuery
                    query_params = ResearchQuery(**request.parameters)
                    results = self.agent.research(query_params.query)
                    
                    return AgentResponse(
                        request_id=request.request_id,
                        status="success",
                        result=ResearchResult(
                            query=query_params.query,
                            summary=results.get('summary', ''),
                            total_papers=results.get('total_papers', 0),
                            total_models=results.get('total_models', 0),
                            arxiv_results=results.get('arxiv_results', []),
                            pubmed_results=results.get('pubmed_results', []),
                            huggingface_results=results.get('huggingface_results', [])
                        ).model_dump(),
                        metadata={"capability": request.capability}
                    )
                
                elif request.capability == "search_rag":
                    # Convert parameters to RAGSearchQuery
                    query_params = RAGSearchQuery(**request.parameters)
                    results = self.agent.search_rag(
                        query=query_params.query,
                        k=query_params.k,
                        source_filter=query_params.source_filter
                    )
                    
                    return AgentResponse(
                        request_id=request.request_id,
                        status="success",
                        result=results,
                        metadata={"capability": request.capability}
                    )
                
                else:
                    return AgentResponse(
                        request_id=request.request_id,
                        status="error",
                        error=f"Unknown capability: {request.capability}",
                        metadata={"requested_capability": request.capability}
                    )
            
            except Exception as e:
                return AgentResponse(
                    request_id=request.request_id,
                    status="error",
                    error=str(e),
                    metadata={"exception_type": type(e).__name__}
                )
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring."""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "rag_enabled": self.agent.enable_rag,
                "llm_available": self.agent.llm is not None
            }
        
        return app
    
    def run(self, **kwargs):
        """
        Run the server.
        
        Args:
            **kwargs: Additional arguments to pass to uvicorn.run()
        """
        config = {
            "host": self.host,
            "port": self.port,
            "log_level": "info"
        }
        config.update(kwargs)
        
        uvicorn.run(self.app, **config)
    
    async def run_async(self, **kwargs):
        """
        Run the server asynchronously.
        
        Useful for embedding the server in an async application.
        """
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_agent_server(
    llm_provider: str = "github",
    host: str = "0.0.0.0",
    port: int = 8000,
    **agent_kwargs
) -> AgentServer:
    """
    Factory function to create an AgentServer with a configured agent.
    
    Args:
        llm_provider: LLM provider to use
        host: Host to bind to
        port: Port to bind to
        **agent_kwargs: Additional arguments for MedicalCVResearchAgent
    
    Returns:
        Configured AgentServer instance
    """
    agent = MedicalCVResearchAgent(llm_provider=llm_provider, **agent_kwargs)
    return AgentServer(agent=agent, host=host, port=port)
