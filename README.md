# Medical Computer Vision Research Agent

An AI-powered agent that summarizes capabilities and uses of available AI models for medical computer vision from scientific research sources and model collections.

## Overview

This agent uses **LangChain**, **LangGraph**, and **MCP (Model Context Protocol)** to automatically search and summarize medical computer vision AI models from:

- **arXiv** - Scientific preprints and papers
- **PubMed** - Biomedical literature database
- **HuggingFace** - Pre-trained AI model collections

## Features

- 🔍 **Multi-Source Search**: Automatically searches arXiv, PubMed, and HuggingFace
- 🤖 **Multiple LLM Providers**: Choose between OpenAI or GitHub Models (free with GitHub Pro!)
- 📊 **Structured Workflow**: LangGraph orchestrates the research workflow
- 🔌 **MCP Tools**: Modular tools for each research source
- 💻 **CLI & Python API**: Use via command line or integrate into your code
- 📚 **RAG Storage**: ChromaDB-powered vector database for storing and searching research results
- 🔎 **Semantic Search**: Find relevant research across all sources using similarity search
- 💰 **Cost-Effective**: Use free GitHub Models or skip AI summarization entirely

## Installation

### Prerequisites

- Python 3.12 or higher
- UV (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or see [UV installation](https://docs.astral.sh/uv/getting-started/installation/))
- **One of the following** (optional for results collection without AI summary):
  - OpenAI API key (pay-per-use)
  - GitHub Pro account + GitHub token (free AI models!)
  - Or run without AI summarization (free)

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

**Option A: OpenAI (Pay-per-use)**
```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_openai_key_here
# Get key at: https://platform.openai.com/api-keys
```

**Option B: GitHub Models (Free with GitHub Pro)**
```bash
# Set GitHub token
export GITHUB_TOKEN=your_github_token_here
# Get token at: https://github.com/settings/tokens
# Requires GitHub Pro subscription for free model access
```

**Option C: No AI Summarization (Free)**
```bash
# No setup needed - use --llm-provider none or --no-summary
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
# OpenAI (pay-per-use)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider openai --model gpt-3.5-turbo

# GitHub Models (free with GitHub Pro)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider github --model gpt-4o

# No AI summarization (free)
uv run python -m research_viz_agent.cli "lung cancer detection" --llm-provider none
```

#### GitHub Models Examples (Free with GitHub Pro!)

```bash
# Use free GPT-4o via GitHub Models
uv run python -m research_viz_agent.cli "skin lesion classification" --llm-provider github --model gpt-4o

# Use Meta Llama models
uv run python -m research_viz_agent.cli "brain tumor MRI" --llm-provider github --model Llama-3.2-11B-Vision-Instruct

# Use Microsoft Phi models
uv run python -m research_viz_agent.cli "chest x-ray AI" --llm-provider github --model Phi-3.5-mini-instruct

# Use Mistral models
uv run python -m research_viz_agent.cli "medical imaging" --llm-provider github --model Mistral-large-2407
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
# Basic usage (defaults to OpenAI)
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

# Option 1: OpenAI (pay-per-use)
agent = MedicalCVResearchAgent(
    llm_provider="openai",
    model_name="gpt-3.5-turbo",
    pubmed_email="your-email@example.com"
)

# Option 2: GitHub Models (free with GitHub Pro)
agent = MedicalCVResearchAgent(
    llm_provider="github",
    model_name="gpt-4o",  # Free GPT-4o via GitHub!
    pubmed_email="your-email@example.com"
)

# Option 3: No AI summarization (free)
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

# Install the package in editable mode
uv pip install -e .
```

### Why UV?

- **Fast**: 10-100x faster than pip and poetry
- **Reliable**: Deterministic resolution with lockfiles
- **Simple**: No separate virtual environment management needed
- **Compatible**: Works with existing Python packaging standards

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for AI summarization - Your OpenAI API key
- `HUGGINGFACE_TOKEN`: Optional - HuggingFace API token for authenticated requests

### Handling OpenAI API Issues

If you encounter quota or billing issues:

```bash
# Check your OpenAI setup
uv run python scripts/check_openai.py

# Run without AI summarization (no OpenAI API calls)
uv run python -m research_viz_agent.cli "lung cancer detection" --no-summary

# Search existing RAG database (no API calls needed)
uv run python -m research_viz_agent.cli --rag-search "deep learning"

# Check what's in your RAG database
uv run python -m research_viz_agent.cli --rag-stats
```

**Common Solutions:**
- **Quota Exceeded**: Add billing info at [OpenAI Billing](https://platform.openai.com/account/billing)
- **Rate Limits**: Wait a moment and retry
- **Invalid Key**: Generate new key at [OpenAI API Keys](https://platform.openai.com/api-keys)

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

# Filter by source
uv run python -m research_viz_agent.cli --rag-search "CNN" --rag-source arxiv

# Get database statistics
uv run python -m research_viz_agent.cli --rag-stats

# Disable RAG for a query
uv run python -m research_viz_agent.cli "query" --no-rag
```

### RAG Database Structure

The ChromaDB collection stores:
- **Papers**: ArXiv and PubMed research papers with abstracts, authors, and metadata
- **Models**: HuggingFace models with descriptions, tags, and usage statistics
- **Metadata**: Source information, URLs, publication dates, and search queries
- **Embeddings**: OpenAI embeddings for semantic similarity search

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
