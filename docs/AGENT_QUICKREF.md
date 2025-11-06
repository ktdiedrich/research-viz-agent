# Agent-to-Agent Communication - Quick Reference

## 🚀 Starting the Server

```bash
# Default (port 8000, GitHub Models)
research-viz-agent serve

# Custom port and host
research-viz-agent serve --port 8080 --host 0.0.0.0

# With specific provider
research-viz-agent serve --llm-provider openai --model gpt-4o-mini
```

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Basic info |
| `/status` | GET | Agent capabilities |
| `/research` | POST | Perform research |
| `/rag/search` | POST | Search RAG |
| `/agent/request` | POST | Generic request |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

## 💻 Client Usage (Python)

### Synchronous

```python
from research_viz_agent.agent_protocol.client import AgentClient

with AgentClient("http://localhost:8000") as client:
    # Research
    result = client.research("lung cancer detection", max_results=20)
    
    # RAG search
    rag = client.search_rag("deep learning", k=10)
    
    # Status
    status = client.get_status()
```

### Asynchronous

```python
import asyncio
from research_viz_agent.agent_protocol.client import AgentClient

async def main():
    async with AgentClient("http://localhost:8000") as client:
        # Concurrent requests
        tasks = [
            client.research_async("topic 1"),
            client.research_async("topic 2")
        ]
        results = await asyncio.gather(*tasks)

asyncio.run(main())
```

## 🔧 cURL Examples

### Get Status
```bash
curl http://localhost:8000/status
```

### Research
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "brain tumor segmentation", "max_results": 10}'
```

### RAG Search
```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "medical imaging AI", "k": 5}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## 🐳 Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["research-viz-agent", "serve"]
```

```bash
docker build -t research-viz-agent .
docker run -p 8000:8000 -e GITHUB_TOKEN=$GITHUB_TOKEN research-viz-agent
```

## 📦 Message Schemas

### Request
```python
{
    "request_id": "req-123",
    "capability": "research_medical_cv",
    "parameters": {"query": "...", "max_results": 20},
    "timeout": 300
}
```

### Response
```python
{
    "request_id": "req-123",
    "status": "success",  # or "error", "partial", "timeout"
    "result": {...},
    "metadata": {},
    "timestamp": "2025-01-01T00:00:00Z"
}
```

## 🔐 Security

### Add Authentication
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(credentials = Depends(security)):
    if credentials.credentials != "secret-token":
        raise HTTPException(401, "Invalid token")
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/research")
@limiter.limit("10/minute")
async def research(request, query):
    ...
```

## 📊 Monitoring

```python
from prometheus_client import Counter, Histogram

requests_total = Counter('requests_total', 'Total requests')
duration = Histogram('request_duration_seconds', 'Request duration')

@app.post("/research")
async def research(query):
    requests_total.inc()
    with duration.time():
        # ... research
```

## 🔗 Related Files

- **Full Guide**: `docs/AGENT_COMMUNICATION.md`
- **Examples**: `examples/agent_*.py`
- **Tests**: `tests/test_agent_protocol.py`
- **Server**: `research_viz_agent/agent_protocol/server.py`
- **Client**: `research_viz_agent/agent_protocol/client.py`
- **Schemas**: `research_viz_agent/agent_protocol/schemas.py`
