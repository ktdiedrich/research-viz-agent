# Medical Computer Vision Research Agent

An AI-powered agent that summarizes capabilities and uses of available AI models for medical computer vision from scientific research sources and model collections.

## Overview

This agent uses **LangChain**, **LangGraph**, and **MCP (Model Context Protocol)** to automatically search and summarize medical computer vision AI models from:

- **arXiv** - Scientific preprints and papers
- **PubMed** - Biomedical literature database
- **HuggingFace** - Pre-trained AI model collections

## Features

- 🔍 **Multi-Source Search**: Automatically searches arXiv, PubMed, and HuggingFace
- 🤖 **AI-Powered Summarization**: Uses LangChain and GPT models to generate comprehensive summaries
- 📊 **Structured Workflow**: LangGraph orchestrates the research workflow
- 🔌 **MCP Tools**: Modular tools for each research source
- 💻 **CLI & Python API**: Use via command line or integrate into your code

## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (install via `pip install poetry` or see [Poetry installation](https://python-poetry.org/docs/#installation))
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ktdiedrich/research-viz-agent.git
cd research-viz-agent
```

2. Install dependencies with Poetry:
```bash
poetry install
```

Or for development with additional tools:
```bash
poetry install --with dev
```

Alternatively, install with pip:
```bash
pip install -e .
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Command Line Interface

```bash
# Basic usage
python -m research_viz_agent.cli "lung cancer detection"

# With custom email for PubMed
python -m research_viz_agent.cli "skin lesion classification" --email your@email.com

# Save results to file
python -m research_viz_agent.cli "brain tumor segmentation" --output results.txt

# Use GPT-4
python -m research_viz_agent.cli "retinal disease detection" --model gpt-4
```

### Python API

```python
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent

# Initialize the agent
agent = MedicalCVResearchAgent(
    pubmed_email="your-email@example.com",
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Run research
results = agent.research("lung cancer detection")

# Display formatted results
print(agent.format_results(results))

# Access individual components
print(f"Found {results['total_papers']} papers")
print(f"Found {results['total_models']} models")
print(f"Summary: {results['summary']}")
```

### Example Script

Run the included example:
```bash
python examples/example_usage.py
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
│   └── cli.py              # Command-line interface
└── examples/               # Example usage scripts
```

### Workflow

1. **Query Processing**: User provides a medical CV research query
2. **Parallel Search**: MCP tools search arXiv, PubMed, and HuggingFace
3. **Data Aggregation**: Results are collected and structured
4. **AI Summarization**: LangChain generates a comprehensive summary
5. **Output Formatting**: Results are formatted for display

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `HUGGINGFACE_TOKEN`: Optional - HuggingFace API token for authenticated requests

### Agent Parameters

- `model_name`: OpenAI model to use (default: `gpt-3.5-turbo`)
- `temperature`: LLM temperature for creativity (default: `0.7`)
- `pubmed_email`: Email for PubMed API (required by NCBI)

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
