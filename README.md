# Medical Computer Vision Research Agent

An AI-powered agent that summarizes capabilities and uses of available AI models for medical computer vision from scientific research sources and model collections.

## Overview

This agent uses **LangChain**, **LangGraph**, and **MCP (Model Context Protocol)** to automatically search and summarize medical computer vision AI models from:

- **arXiv** - Scientific preprints and papers
- **PubMed** - Biomedical literature database
- **HuggingFace** - Pre-trained AI model collections

**✨ NEW**: Now supports **GitHub Models** as the default provider, offering **free AI models** (GPT-4o, Llama, Phi, Mistral) for users with GitHub Pro subscriptions!

## Features

- 🔍 **Multi-Source Search**: Automatically searches arXiv, PubMed, and HuggingFace
- 🆓 **GitHub Models Integration**: Free AI models (GPT-4o, Llama, Phi, Mistral) with GitHub Pro subscription
- 🤖 **Multiple LLM Providers**: Choose between GitHub Models (default), OpenAI, or no AI
- 📊 **Structured Workflow**: LangGraph orchestrates the research workflow
- 🔌 **MCP Tools**: Modular tools for each research source
- 💻 **CLI & Python API**: Use via command line or integrate into your code
- 📚 **RAG Storage**: Provider-specific ChromaDB vector databases with matching embeddings
- 🔎 **Semantic Search**: Find relevant research using GitHub or OpenAI embeddings
- 📊 **Query Tracking**: Automatic visualization of queries and records added to RAG store
- 💰 **Cost-Effective**: Completely free with GitHub Pro, or pay-per-use with OpenAI
- 🔗 **Agent-to-Agent Communication**: REST API server for inter-agent interaction
- 📤 **CSV Export**: Export research and RAG results to CSV format
- 📈 **RAG Visualization**: Visualize embeddings and clustering with t-SNE/UMAP

## Installation

### Prerequisites

- Python 3.12 or higher
- UV (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or see [UV installation](https://docs.astral.sh/uv/getting-started/installation/))
- **Choose your AI provider** (optional - can run without AI for data collection only):
  - **🆓 GitHub Pro + GitHub token** (recommended - free AI models!)
  - OpenAI API key (pay-per-use alternative)
  - Or run without AI summarization (free data collection)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ktdiedrich/research-viz-agent.git
cd research-viz-agent
```

2. Install dependencies with UV:
```bash
uv sync
```

Or for development with additional tools:
```bash
uv sync --extra dev
```

3. Configure API keys (choose one option):

**Option A: GitHub Models (Recommended - Free with GitHub Pro) 🆓**
```bash
# Set GitHub token for free AI models
export GITHUB_TOKEN=your_github_token_here
# Get token at: https://github.com/settings/tokens
# Requires GitHub Pro subscription for free access to GPT-4o, Llama, Phi, Mistral models
```

**Option B: OpenAI (Pay-per-use)**
```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_openai_key_here
# Get key at: https://platform.openai.com/api-keys
```

**Option C: No AI Summarization (Free data collection only)**
```bash
# No setup needed - use --llm-provider none
# Collects research data without AI summarization
```

4. Activate the UV virtual environment:
```bash
# UV automatically manages the virtual environment
# To run commands in the environment, prefix with 'uv run'
# or activate manually:
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Usage

### Command Line Interface

#### Basic Usage with Different LLM Providers

```bash
# GitHub Models (default - free with GitHub Pro) 🆓
uv run python -m research_viz_agent.cli "lung cancer detection"
# Uses GitHub's free GPT-4o model by default

# Specify different GitHub model
uv run python -m research_viz_agent.cli "lung cancer detection" --model Llama-3.2-11B-Vision-Instruct

# OpenAI (pay-per-use alternative)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider openai --model gpt-3.5-turbo

# No AI summarization (free data collection)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider none
```

#### GitHub Models Examples (Default Provider - Free with GitHub Pro!) 🆓

```bash
# Use free GPT-4o (default model)
uv run python -m research_viz_agent.cli "skin lesion classification"

# Use free GPT-4o-mini (faster)
uv run python -m research_viz_agent.cli "brain tumor MRI" --model gpt-4o-mini

# Use Meta Llama models with vision capabilities
uv run python -m research_viz_agent.cli "chest x-ray AI" --model Llama-3.2-11B-Vision-Instruct

# Use Microsoft Phi models (efficient)
uv run python -m research_viz_agent.cli "heart disease detection" --model Phi-3.5-mini-instruct

# Use Mistral models (multilingual support)
uv run python -m research_viz_agent.cli "medical imaging" --model Mistral-large-2407
```

#### Provider Information and Setup

```bash
# List available models for each provider
uv run python -m research_viz_agent.cli --list-models openai
uv run python -m research_viz_agent.cli --list-models github

# Get provider setup information
uv run python -m research_viz_agent.cli --provider-info github
uv run python -m research_viz_agent.cli --provider-info openai

# Test provider configurations
uv run python scripts/check_llm_providers.py
```

#### General Usage

```bash
# Basic usage (defaults to GitHub Models - free with GitHub Pro)
uv run python -m research_viz_agent.cli "lung cancer detection"

# With custom options
uv run python -m research_viz_agent.cli "skin lesion classification" --email your@email.com --display-results 10

# Save results to file
uv run python -m research_viz_agent.cli "brain tumor segmentation" --output results.txt

# Search the RAG database (after building it with previous queries)
uv run python -m research_viz_agent.cli --rag-search "deep learning medical imaging"

# Search specific source in RAG
uv run python -m research_viz_agent.cli --rag-search "CNN chest x-ray" --rag-source arxiv

# Show RAG database statistics
uv run python -m research_viz_agent.cli --rag-stats

# Disable RAG functionality
uv run python -m research_viz_agent.cli "lung cancer detection" --no-rag
```

### Python API

```python
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent

# Option 1: GitHub Models (default - free with GitHub Pro) 🆓
agent = MedicalCVResearchAgent(
    # Uses GitHub provider by default
    model_name="gpt-4o",  # Free GPT-4o via GitHub Models
    pubmed_email="your-email@example.com"
)

# Option 2: Explicit GitHub Models with different model
agent = MedicalCVResearchAgent(
    llm_provider="github",
    model_name="Llama-3.2-11B-Vision-Instruct",  # Free Meta Llama model
    pubmed_email="your-email@example.com"
)

# Option 3: OpenAI (pay-per-use alternative)
agent = MedicalCVResearchAgent(
    llm_provider="openai",
    model_name="gpt-3.5-turbo",
    pubmed_email="your-email@example.com"
)

# Option 4: No AI summarization (free data collection)
agent = MedicalCVResearchAgent(
    llm_provider="none",
    pubmed_email="your-email@example.com"
)

# Run research
results = agent.research("lung cancer detection")

# Display formatted results
print(agent.format_results(results))

# Access individual components
print(f"Found {results['total_papers']} papers")
print(f"Found {results['total_models']} models")
print(f"Summary: {results['summary']}")

# Search the RAG database
rag_results = agent.search_rag("deep learning medical imaging", k=10)
print(agent.format_rag_results(rag_results))

# Get RAG statistics
stats = agent.get_rag_stats()
print(f"RAG database contains {stats['document_count']} documents")
```

### Example Scripts

Run the included examples:
```bash
# Basic usage example
uv run python examples/example_usage.py

# RAG functionality demonstration
uv run python examples/rag_demo.py

# GitHub Models integration demo
uv run python examples/github_models_demo.py

# Test all LLM providers
uv run python scripts/check_llm_providers.py
```

## Architecture

The agent follows a modular architecture:

```
research-viz-agent/
├── research_viz_agent/
│   ├── mcp_tools/          # MCP tools for each source
│   │   ├── arxiv_tool.py
│   │   ├── pubmed_tool.py
│   │   └── huggingface_tool.py
│   ├── agents/             # LangGraph workflows and agents
│   │   ├── research_workflow.py
│   │   └── medical_cv_agent.py
│   ├── utils/              # Utility functions
│   │   └── rag_store.py    # ChromaDB RAG storage
│   └── cli.py              # Command-line interface
└── examples/               # Example usage scripts
```

### Workflow

1. **Query Processing**: User provides a medical CV research query
2. **Parallel Search**: MCP tools search arXiv, PubMed, and HuggingFace
3. **Data Aggregation**: Results are collected and structured
4. **RAG Storage**: Results are indexed in ChromaDB for future semantic search
5. **AI Summarization**: LangChain generates a comprehensive summary
6. **Output Formatting**: Results are formatted for display

## UV Package Management

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python package management and project setup.

### Key UV Commands

```bash
# Install dependencies and create virtual environment
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Run a command in the UV environment
uv run python script.py

# Run tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=research_viz_agent --cov-report=term-missing

# Install the package in editable mode
uv pip install -e .
```

### Why UV?

- **Fast**: 10-100x faster than pip and poetry
- **Reliable**: Deterministic resolution with lockfiles
- **Simple**: No separate virtual environment management needed
- **Compatible**: Works with existing Python packaging standards

## Testing

The project includes comprehensive test coverage with 263 tests covering all major components.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=research_viz_agent --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_cli.py

# Run tests with verbose output
uv run pytest -v
```

### Test Coverage

The project maintains **85% overall test coverage** across:

- **MCP Tools** (100%): ArXiv, PubMed, and HuggingFace integrations
- **LLM Factory** (100%): Provider management and model configuration
- **Research Workflow** (98%): LangGraph workflow orchestration
- **Medical CV Agent** (93%): Core agent functionality
- **RAG Store** (79%): Vector database operations
- **CLI** (78%): Command-line interface

### Coverage Report Details

The `--cov-report=term-missing` flag shows:
- **Stmts**: Total number of code statements
- **Miss**: Number of statements not covered by tests
- **Branch**: Number of conditional branches
- **BrPart**: Partially covered branches
- **Cover**: Percentage of code coverage
- **Missing**: Line numbers of uncovered code

Example output:
```
Name                                    Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------------------------
research_viz_agent/mcp_tools/arxiv_tool.py    52      0     10      0   100%
research_viz_agent/utils/llm_factory.py       78      0     40      0   100%
research_viz_agent/cli.py                    127     27     44      5    78%   214-243, 315
------------------------------------------------------------------------------------
TOTAL                                        1006    139    304     24    85%
```

### Test Organization

Tests are organized by module:
- `test_arxiv_tool.py`: ArXiv search functionality
- `test_pubmed_tool.py`: PubMed search functionality
- `test_huggingface_tool.py`: HuggingFace model search
- `test_llm_factory.py`: LLM provider management
- `test_medical_cv_agent.py`: Main agent functionality
- `test_research_workflow.py`: LangGraph workflow
- `test_rag_store.py`: Vector database operations
- `test_cli.py`: Command-line interface

## Configuration

### Environment Variables

**LLM Providers (choose one):**
- `GITHUB_TOKEN`: For GitHub Models (free with GitHub Pro subscription) 🆓 **Recommended**
- `OPENAI_API_KEY`: For OpenAI models (pay-per-use alternative)

**Optional:**
- `HUGGINGFACE_TOKEN`: HuggingFace API token for authenticated requests

### LLM Provider Comparison

| Provider | Cost | Setup | Models Available | Notes |
|----------|------|-------|------------------|-------|
| **GitHub Models** 🆓 | **FREE** | [Get Token](https://github.com/settings/tokens) | GPT-4o, Llama, Phi, Mistral | **Recommended**: GitHub Pro subscription |
| **OpenAI** | Pay-per-use | [Get API Key](https://platform.openai.com/api-keys) | GPT-3.5, GPT-4, GPT-4o | Alternative: Official OpenAI API |
| **None** | **FREE** | No setup | N/A | Data collection only (no AI summary) |

### Handling API Issues

If you encounter provider or API issues:

```bash
# Test your provider setup
uv run python scripts/check_llm_providers.py

# Switch to free GitHub Models (recommended)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider github

# Run without AI summarization (completely free)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider none

# Search existing RAG database (no API calls needed)
uv run python -m research_viz_agent.cli --rag-search "deep learning"

# Check what's in your RAG database
uv run python -m research_viz_agent.cli --rag-stats
```

**Common Solutions:**
- **GitHub Models**: Ensure you have GitHub Pro subscription and valid token
- **OpenAI Issues**: Check billing at [OpenAI Billing](https://platform.openai.com/account/billing)
- **Rate Limits**: Wait a moment and retry, or switch to GitHub Models (free)
- **Invalid Tokens**: Generate new tokens at provider settings

### Agent Parameters

- `model_name`: OpenAI model to use (default: `gpt-3.5-turbo`)
- `temperature`: LLM temperature for creativity (default: `0.7`)
- `pubmed_email`: Email for PubMed API (required by NCBI)
- `enable_rag`: Enable RAG storage and search (default: `True`)
- `rag_persist_dir`: Directory for ChromaDB storage (default: `./chroma_db`)

## RAG (Retrieval-Augmented Generation) Features

The agent includes a powerful RAG system using ChromaDB for persistent storage and semantic search of research results.

### RAG Benefits

- **Persistent Storage**: All research results are stored locally for future access
- **Semantic Search**: Find relevant papers/models using natural language queries
- **Cross-Query Analysis**: Search across results from multiple research sessions
- **Offline Capability**: Search stored results without making new API calls
- **Source Filtering**: Filter searches by source (ArXiv, PubMed, HuggingFace)

### RAG Usage

```bash
# Build RAG database through research
uv run python -m research_viz_agent.cli "lung cancer detection"
uv run python -m research_viz_agent.cli "skin lesion classification"

# Search the RAG database
uv run python -m research_viz_agent.cli --rag-search "deep learning medical imaging"

# Export RAG search results to CSV
uv run python -m research_viz_agent.cli --rag-search "deep learning medical imaging" --csv results.csv

# Export with more results and source filtering
uv run python -m research_viz_agent.cli --rag-search "CNN" --rag-source arxiv --rag-results 50 --csv arxiv_cnn.csv

# Get database statistics
uv run python -m research_viz_agent.cli --rag-stats

# Disable RAG for a query
uv run python -m research_viz_agent.cli "query" --no-rag
```

### Agent-to-Agent Communication

Start the agent as an HTTP server for agent-to-agent communication:

```bash
# Start server with default settings (port 8000)
uv run python -m research_viz_agent.cli serve

# Customize server configuration
uv run python -m research_viz_agent.cli serve --port 8080 --host 0.0.0.0

# Start with specific LLM provider
uv run python -m research_viz_agent.cli serve --llm-provider openai --model gpt-4o-mini
```

**Access the API:**
- Base URL: `http://localhost:8000`
- Status: `http://localhost:8000/status`
- Interactive Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

**Use the client to communicate with the server:**

```python
from research_viz_agent.agent_protocol.client import AgentClient

# Connect to the agent
with AgentClient(base_url="http://localhost:8000") as client:
    # Check agent status
    status = client.get_status()
    print(f"Agent: {status.agent_name} v{status.version}")
    
    # Perform research
    result = client.research(
        query="lung cancer detection",
        max_results=20
    )
    print(f"Found {result.total_papers} papers")
    
    # Search RAG database
    rag_results = client.search_rag(query="CNN medical imaging", k=10)
    print(f"Found {rag_results.total_count} documents")
```

See [Agent Communication Guide](docs/AGENT_COMMUNICATION.md) for full documentation including:
- Agent orchestration examples
- Multi-agent workflows
- Deployment (Docker, Kubernetes)
- Security and authentication
- API reference

### CSV Export

Export research results and RAG searches to CSV format for analysis in spreadsheet applications:

```bash
# Export regular research results
uv run python -m research_viz_agent.cli "lung cancer detection" --csv research.csv

# Export RAG search results
uv run python -m research_viz_agent.cli --rag-search "medical imaging AI" --csv search.csv

# Export with specific source filter
uv run python -m research_viz_agent.cli --rag-search "deep learning" --rag-source pubmed --csv pubmed_papers.csv
```

**CSV columns include:**
- Source (arxiv, pubmed, huggingface)
- Type (paper, model)
- Title, Authors, Abstract
- URL, Publication Date
- Source-specific metadata (PMID, categories, tags, downloads, etc.)
- Query used to find the document

### RAG Database Structure

The ChromaDB collection stores:
- **Papers**: ArXiv and PubMed research papers with abstracts, authors, and metadata
- **Models**: HuggingFace models with descriptions, tags, and usage statistics
- **Metadata**: Source information, URLs, publication dates, and search queries
- **Embeddings**: Provider-specific embeddings (GitHub Models or OpenAI) for semantic similarity search

### RAG Embedding Visualization & Clustering Analysis

Visualize how documents are encoded and clustered in the vector database:

```bash
# Basic 2D visualization
uv run python scripts/visualize_rag_embeddings.py --output rag_viz.png

# With clustering analysis
uv run python scripts/visualize_rag_embeddings.py --cluster --output rag_clustered.png

# Interactive HTML report
uv run python scripts/visualize_rag_embeddings.py --cluster --html rag_report.html

# 3D visualization with UMAP
uv run python scripts/visualize_rag_embeddings.py --method umap --3d --output rag_3d.png

# Run complete demo
uv run python examples/rag_embeddings_demo.py
```

**Features:**
- 📊 2D/3D visualization of document embeddings
- 🔍 Automatic clustering analysis (k-means, DBSCAN)
- 📈 Silhouette scores and cluster quality metrics
- 🎨 Color-coded by source (ArXiv, PubMed, HuggingFace)
- 🌐 Interactive HTML reports with hover details
- 📉 Dimensionality reduction (t-SNE, UMAP)

See [docs/RAG_ENCODING_CLUSTERING.md](docs/RAG_ENCODING_CLUSTERING.md) for detailed information about how documents are encoded, stored, and clustered in the vector database.

### RAG Query Tracking & Visualization

The system automatically tracks and visualizes all queries and records added to the RAG store:

```bash
# View tracking chart in terminal
uv run python -m research_viz_agent.cli --show-tracking

# Show summary statistics
uv run python -m research_viz_agent.cli --tracking-summary

# Generate interactive HTML chart
uv run python scripts/visualize_rag_tracking.py --html rag_chart.html

# Show recent queries only
uv run python scripts/visualize_rag_tracking.py --recent 5

# Run tracking demo
uv run python examples/rag_tracking_demo.py
```

**Features:**
- 📊 Bar charts showing records added per query
- 🕐 Timestamps for each query
- 📈 Breakdown by source (ArXiv, PubMed, HuggingFace)
- 💾 Automatic tracking of all additions
- 🌐 HTML and ASCII chart formats

See [docs/RAG_TRACKING.md](docs/RAG_TRACKING.md) for detailed documentation.

## Example Output

```
================================================================================
RESEARCH SUMMARY: lung cancer detection
================================================================================

Total Papers Found: 30
Total Models Found: 15

--------------------------------------------------------------------------------
AI-GENERATED SUMMARY
--------------------------------------------------------------------------------

[Comprehensive summary of AI models, capabilities, applications, trends, and recommendations]

--------------------------------------------------------------------------------
DETAILED SOURCES
--------------------------------------------------------------------------------

### ArXiv Papers ###

1. Deep Learning for Lung Cancer Detection in CT Scans
   URL: https://arxiv.org/abs/...
   Published: 2023-05-15

[Additional papers and models...]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph)
- Research sources: [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [HuggingFace](https://huggingface.co/)

## Support

For issues and questions, please open an issue on GitHub. 
