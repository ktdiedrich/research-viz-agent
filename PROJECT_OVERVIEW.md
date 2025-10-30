# Medical CV Research Agent - Project Overview

## 🎯 Purpose

AI agent that automatically researches and summarizes medical computer vision AI models by searching:
- 📚 **arXiv** - Academic papers
- 🏥 **PubMed** - Medical literature  
- 🤖 **HuggingFace** - Pre-trained models

## ⚡ Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure (add your OpenAI API key)
cp .env.example .env

# Run
python -m research_viz_agent.cli "lung cancer detection"
```

## 📦 What's Included

### Core Components
✅ **MCP Tools** - arXiv, PubMed, HuggingFace integrations  
✅ **LangGraph Workflow** - State machine for orchestration  
✅ **LangChain Agent** - LLM-powered summarization  
✅ **CLI & Python API** - Multiple interfaces  

### Documentation
✅ README.md - Main documentation  
✅ docs/QUICKSTART.md - 5-minute guide  
✅ docs/API.md - API reference  
✅ docs/CONFIGURATION.md - Setup guide  
✅ docs/ARCHITECTURE.md - System design  
✅ CONTRIBUTING.md - Developer guide  

### Testing
✅ Unit tests for components  
✅ Standalone tool tests  
✅ Example usage scripts  

## 🏗️ Architecture

```
User → CLI/API → MedicalCVResearchAgent
                      ↓
              LangGraph Workflow
                      ↓
        ┌─────────┬──────────┬────────────┐
        │  arXiv  │  PubMed  │ HuggingFace│
        └─────────┴──────────┴────────────┘
                      ↓
            LLM Summarization
                      ↓
            Formatted Results
```

## 💡 Example Usage

### Command Line
```bash
python -m research_viz_agent.cli "brain tumor segmentation" \
  --email your@email.com \
  --output results.txt
```

### Python
```python
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent

agent = MedicalCVResearchAgent(pubmed_email="your@email.com")
results = agent.research("lung cancer detection")
print(agent.format_results(results))
```

## 📊 Output Includes

1. **AI-Generated Summary** - Comprehensive analysis
2. **Papers** - Relevant research from arXiv & PubMed
3. **Models** - Available models from HuggingFace
4. **Insights** - Key trends and recommendations
5. **Links** - Direct links to all sources

## 🔧 Technology

- **LangChain** - LLM orchestration
- **LangGraph** - Workflow management
- **MCP** - Tool integration
- **OpenAI GPT** - Summarization
- **Python 3.9+** - Implementation

## 📝 Files Overview

```
research-viz-agent/
├── research_viz_agent/       # Main package
│   ├── mcp_tools/            # Research source integrations
│   ├── agents/               # Agent & workflow logic
│   └── cli.py                # Command-line interface
├── examples/                 # Example scripts
├── tests/                    # Test suite
├── docs/                     # Documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # Main docs
```

## 🚀 Next Steps

1. Read [QUICKSTART.md](docs/QUICKSTART.md) to get started
2. See [API.md](docs/API.md) for detailed API docs
3. Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details
4. Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## 📧 Support

- Read the documentation in `docs/`
- Check existing GitHub issues
- Open a new issue for bugs/features

---

**Ready to use!** Just add your OpenAI API key and start researching. 🔬🤖
