# Agent-to-Agent Communication Guide

## Overview

The research-viz-agent now supports **agent-to-agent communication**, enabling it to function as a service that other AI agents can interact with programmatically. This transforms the agent from a standalone CLI tool into a **releasable, production-ready service**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Agent Communication Layer                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         HTTP/REST API       ┌──────────┐ │
│  │  External    │ ─────────────────────────▶  │  Agent   │ │
│  │  Agent       │                              │  Server  │ │
│  │  (Client)    │ ◀─────────────────────────  │          │ │
│  └──────────────┘    JSON Request/Response    └──────────┘ │
│                                                       │      │
│                                                       ▼      │
│                                         ┌─────────────────┐ │
│                                         │ Research Agent  │ │
│                                         │   - ArXiv       │ │
│                                         │   - PubMed      │ │
│                                         │   - HuggingFace │ │
│                                         │   - RAG Store   │ │
│                                         └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Agent Protocol (`research_viz_agent/agent_protocol/`)

#### Schemas (`schemas.py`)
- **AgentRequest** - Standardized request format
- **AgentResponse** - Standardized response format
- **AgentStatus** - Agent capabilities and status
- **ResearchQuery** - Research request parameters
- **ResearchResult** - Research response data
- **RAGSearchQuery** - RAG search parameters
- **RAGSearchResult** - RAG search results

All schemas use **Pydantic** for validation and serialization.

#### Server (`server.py`)
- **AgentServer** - FastAPI-based HTTP server
- Exposes research agent capabilities via REST API
- Supports both sync and async operations
- Includes health checks and status endpoints
- Auto-generated OpenAPI documentation

#### Client (`client.py`)
- **AgentClient** - HTTP client for agent communication
- Provides high-level methods for invoking capabilities
- Supports both sync (`client.research()`) and async (`client.research_async()`)
- Automatic request formatting and response parsing
- Context manager support for connection cleanup

## Quick Start

### Starting the Agent Server

```bash
# Start server with default settings (port 8000, GitHub Models)
research-viz-agent serve

# Customize server configuration
research-viz-agent serve --port 8080 --host 0.0.0.0 --llm-provider openai

# Start programmatically
python examples/agent_server.py
```

### Using the Agent Client

```python
from research_viz_agent.agent_protocol.client import AgentClient

# Connect to the agent
with AgentClient(base_url="http://localhost:8000") as client:
    # Check agent status
    status = client.get_status()
    print(f"Agent: {status.agent_name} v{status.version}")
    
    # Perform research
    result = client.research(
        query="lung cancer detection deep learning",
        max_results=20
    )
    print(f"Found {result.total_papers} papers")
    print(f"Summary: {result.summary}")
    
    # Search RAG database
    rag_results = client.search_rag(
        query="convolutional neural networks",
        k=10
    )
    print(f"Found {rag_results.total_count} relevant documents")
```

## API Endpoints

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic agent info |
| `/status` | GET | Agent status and capabilities |
| `/research` | POST | Perform research query |
| `/rag/search` | POST | Search RAG database |
| `/agent/request` | POST | Generic agent request |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

### Example Requests

#### Get Agent Status
```bash
curl http://localhost:8000/status
```

Response:
```json
{
  "agent_id": "research-viz-agent-001",
  "agent_name": "Medical CV Research Agent",
  "version": "0.1.0",
  "status": "online",
  "capabilities": [
    {
      "name": "research_medical_cv",
      "description": "Search and summarize medical computer vision research"
    },
    {
      "name": "search_rag",
      "description": "Search persistent RAG database"
    }
  ]
}
```

#### Research Request
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "brain tumor segmentation",
    "max_results": 10,
    "enable_rag": true
  }'
```

#### RAG Search
```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning medical imaging",
    "k": 5,
    "source_filter": "arxiv"
  }'
```

## Agent Orchestration Examples

### Multi-Agent Workflow

Create an orchestrator agent that coordinates multiple research tasks:

```python
import asyncio
from research_viz_agent.agent_protocol.client import AgentClient

class OrchestratorAgent:
    def __init__(self):
        self.research_client = AgentClient("http://localhost:8000")
    
    async def research_multiple_topics(self, topics):
        """Research multiple topics concurrently."""
        tasks = [
            self.research_client.research_async(query=topic)
            for topic in topics
        ]
        return await asyncio.gather(*tasks)

# Use the orchestrator
async def main():
    orchestrator = OrchestratorAgent()
    results = await orchestrator.research_multiple_topics([
        "lung cancer detection",
        "brain tumor segmentation",
        "retinal disease classification"
    ])
    print(f"Researched {len(results)} topics concurrently")

asyncio.run(main())
```

See `examples/agent_orchestration.py` for a complete example with:
- Concurrent multi-topic research
- Iterative research using RAG
- Result aggregation

## Message Protocol

### Standard Request Format

```python
from research_viz_agent.agent_protocol.schemas import AgentRequest

request = AgentRequest(
    request_id="req-12345",
    capability="research_medical_cv",
    parameters={
        "query": "medical imaging AI",
        "max_results": 20
    },
    metadata={
        "requesting_agent": "orchestrator-001",
        "priority": "normal"
    },
    timeout=300
)
```

### Standard Response Format

```python
from research_viz_agent.agent_protocol.schemas import AgentResponse

response = AgentResponse(
    request_id="req-12345",
    status="success",  # or "error", "partial", "timeout"
    result={
        "query": "medical imaging AI",
        "summary": "...",
        "total_papers": 45
    },
    metadata={
        "execution_time_ms": 4523,
        "sources_searched": ["arxiv", "pubmed", "huggingface"]
    }
)
```

## Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY research_viz_agent/ ./research_viz_agent/

# Install dependencies
RUN pip install -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV GITHUB_TOKEN=""
ENV OPENAI_API_KEY=""

# Run server
CMD ["research-viz-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t research-viz-agent .
docker run -p 8000:8000 \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  research-viz-agent
```

### Kubernetes Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-viz-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: research-viz-agent
  template:
    metadata:
      labels:
        app: research-viz-agent
    spec:
      containers:
      - name: agent
        image: research-viz-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: github-token
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: research-viz-agent-service
spec:
  selector:
    app: research-viz-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

## Security Considerations

### Authentication

Add authentication to the server:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials
```

### Rate Limiting

Consider adding rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/research")
@limiter.limit("10/minute")
async def research(request: Request, query: ResearchQuery):
    # ... research logic
```

## Testing

Run the agent protocol tests:

```bash
# Run all tests
pytest tests/test_agent_protocol.py -v

# Test server endpoints
pytest tests/test_agent_protocol.py::test_server_status -v

# Test client communication
pytest tests/test_agent_protocol.py::test_client_research -v
```

## Monitoring

### Health Checks

The `/health` endpoint provides status information:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "agent_id": "research-viz-agent-001",
  "rag_enabled": true,
  "llm_available": true
}
```

### Metrics

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

research_requests = Counter('research_requests_total', 'Total research requests')
research_duration = Histogram('research_duration_seconds', 'Research duration')

@app.post("/research")
async def research(query: ResearchQuery):
    research_requests.inc()
    with research_duration.time():
        # ... research logic
```

## Examples

All examples are in the `examples/` directory:

1. **`agent_server.py`** - Start the agent server
2. **`agent_client.py`** - Client usage examples (sync and async)
3. **`agent_orchestration.py`** - Multi-agent workflow

Run examples:
```bash
# Terminal 1: Start server
python examples/agent_server.py

# Terminal 2: Run client
python examples/agent_client.py

# Terminal 3: Run orchestration
python examples/agent_orchestration.py
```

## FAQ

**Q: Can I run multiple agent servers?**
A: Yes, start servers on different ports and use a load balancer.

**Q: How do I handle long-running requests?**
A: Implement async endpoints and return a task ID for status polling.

**Q: Can agents communicate without HTTP?**
A: You can implement other transports (WebSocket, gRPC, message queues) using the same schemas.

**Q: How do I version my agent API?**
A: Add version prefixes to URLs (`/v1/research`, `/v2/research`) or use headers.

**Q: Can I use this with LangChain agents?**
A: Yes! The client can be used as a LangChain tool.

## Related Documentation

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API.md)
- [Configuration Guide](CONFIGURATION.md)
- [Quick Start](QUICKSTART.md)
