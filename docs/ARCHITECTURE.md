# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌──────────────┐             │
│  │  CLI (cli.py)│              │ Python API   │             │
│  └──────┬───────┘              └──────┬───────┘             │
│         │                              │                     │
│         └──────────────┬───────────────┘                     │
│                        │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MedicalCVResearchAgent (main agent)             │
│                 (medical_cv_agent.py)                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  • Orchestrates the research workflow                        │
│  • Manages LLM (OpenAI GPT models)                          │
│  • Formats and presents results                              │
│                                                               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            ResearchWorkflow (LangGraph)                      │
│              (research_workflow.py)                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐           │
│  │  Search   │───▶│  Search   │───▶│  Search   │           │
│  │  arXiv    │    │  PubMed   │    │HuggingFace│           │
│  └───────────┘    └───────────┘    └───────────┘           │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           ▼                                  │
│                    ┌───────────┐                             │
│                    │Summarize  │                             │
│                    │(with LLM) │                             │
│                    └───────────┘                             │
│                                                               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  MCP Tools Layer                             │
│                  (mcp_tools/)                                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  ArxivTool   │  │ PubMedTool   │  │HuggingFace   │      │
│  │              │  │              │  │    Tool      │      │
│  │• Search      │  │• Search      │  │• Search      │      │
│  │  papers      │  │  papers      │  │  models      │      │
│  │• Filter by   │  │• Medical CV  │  │• Get model   │      │
│  │  category    │  │  focus       │  │  info        │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │                │
└─────────┼─────────────────┼─────────────────┼────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  External APIs                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   arXiv      │  │   PubMed     │  │ HuggingFace  │      │
│  │  arxiv.org   │  │ NCBI/PubMed  │  │huggingface.co│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   AI/LLM Layer                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────┐             │
│  │         LangChain + OpenAI GPT             │             │
│  │  • Text generation and summarization       │             │
│  │  • Context understanding                   │             │
│  │  • Prompt engineering                      │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer

**CLI (Command-Line Interface)**
- File: `research_viz_agent/cli.py`
- Provides command-line access to the agent
- Handles argument parsing and output formatting

**Python API**
- File: `examples/example_usage.py`
- Programmatic access to the agent
- Integration into other Python applications

### 2. Agent Layer

**MedicalCVResearchAgent**
- File: `research_viz_agent/agents/medical_cv_agent.py`
- Main entry point for research queries
- Manages configuration and API keys
- Formats and presents results to users

### 3. Workflow Layer

**ResearchWorkflow (LangGraph)**
- File: `research_viz_agent/agents/research_workflow.py`
- Implements the research workflow as a state machine
- Coordinates parallel searches across sources
- Orchestrates LLM summarization

**Workflow Steps:**
1. **search_arxiv**: Query arXiv for academic papers
2. **search_pubmed**: Query PubMed for medical literature
3. **search_huggingface**: Query HuggingFace for AI models
4. **summarize**: Generate comprehensive summary using LLM

### 4. MCP Tools Layer

**ArxivTool**
- File: `research_viz_agent/mcp_tools/arxiv_tool.py`
- Searches arXiv using the arxiv Python library
- Filters for computer vision and medical imaging papers
- Returns paper metadata and abstracts

**PubMedTool**
- File: `research_viz_agent/mcp_tools/pubmed_tool.py`
- Searches PubMed/NCBI using BioPython
- Focuses on medical computer vision research
- Respects NCBI rate limits

**HuggingFaceTool**
- File: `research_viz_agent/mcp_tools/huggingface_tool.py`
- Searches HuggingFace model hub via REST API
- Filters for medical imaging models
- Returns model metadata and statistics

### 5. External APIs

**arXiv**
- Academic preprint repository
- Focus: CS, AI, medical physics
- No authentication required

**PubMed**
- Biomedical literature database
- Maintained by NCBI
- Requires email for API access

**HuggingFace**
- AI model repository
- Community-driven model sharing
- Optional authentication for higher limits

### 6. AI/LLM Layer

**LangChain + OpenAI GPT**
- Handles natural language processing
- Generates comprehensive summaries
- Combines information from multiple sources
- Provides structured, actionable insights

## Data Flow

```
User Query
    │
    ▼
MedicalCVResearchAgent.research()
    │
    ▼
ResearchWorkflow.run()
    │
    ├─▶ ArxivTool.search_medical_cv_models()
    │       └─▶ arXiv API ─▶ Papers
    │
    ├─▶ PubMedTool.search_medical_cv_models()
    │       └─▶ PubMed API ─▶ Papers
    │
    └─▶ HuggingFaceTool.search_medical_cv_models()
            └─▶ HuggingFace API ─▶ Models
    │
    ▼
All Results Aggregated
    │
    ▼
LLM Summarization (via LangChain)
    │
    ▼
Formatted Results
    │
    ▼
User Output (CLI or API)
```

## Technology Stack

### Core Frameworks
- **LangChain**: LLM orchestration and prompt management
- **LangGraph**: Workflow state machine and agent orchestration
- **MCP**: Model Context Protocol for tool integration

### Research APIs
- **arxiv**: Python library for arXiv access
- **BioPython**: PubMed/NCBI access
- **requests**: HuggingFace API calls

### LLM
- **OpenAI GPT-3.5/4**: Text generation and summarization

### Utilities
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation
- **beautifulsoup4**: HTML parsing (if needed)

## Key Design Decisions

### 1. Modular MCP Tools
Each research source has its own independent tool, making it easy to:
- Add new sources
- Modify existing sources
- Test tools independently
- Swap implementations

### 2. LangGraph Workflow
Using LangGraph provides:
- Clear workflow visualization
- State management
- Easy modification of workflow steps
- Parallel execution capability

### 3. Separation of Concerns
- **Tools**: Data retrieval only
- **Workflow**: Orchestration only
- **Agent**: High-level interface only
- **LLM**: Summarization only

### 4. Extensibility
The architecture supports:
- Adding new research sources
- Custom summarization strategies
- Different LLM providers
- Alternative output formats

## Configuration Points

1. **Environment Variables** (`.env`)
   - OPENAI_API_KEY
   - HUGGINGFACE_TOKEN

2. **Agent Configuration**
   - LLM model selection
   - Temperature settings
   - Email for PubMed

3. **Tool Configuration**
   - Max results per source
   - Search filters
   - Rate limiting

## Error Handling

- Network failures: Retry logic in tools
- API rate limits: Automatic delays
- Missing results: Graceful degradation
- Invalid queries: User-friendly error messages

## Performance Considerations

- **Parallel searches**: LangGraph enables concurrent API calls
- **Result caching**: Could be added for repeated queries
- **Rate limiting**: Built-in delays respect API limits
- **Token optimization**: Truncate long abstracts for LLM context
