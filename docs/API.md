# API Documentation

## MedicalCVResearchAgent

The main agent class for researching medical computer vision AI models.

### Initialization

```python
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent

agent = MedicalCVResearchAgent(
    openai_api_key=None,           # Optional: defaults to OPENAI_API_KEY env var
    huggingface_token=None,         # Optional: defaults to HUGGINGFACE_TOKEN env var
    pubmed_email="research@example.com",  # Required for PubMed API
    model_name="gpt-3.5-turbo",    # OpenAI model to use
    temperature=0.7                 # LLM creativity (0.0-1.0)
)
```

### Methods

#### `research(query: str) -> Dict`

Perform comprehensive research on medical CV AI models.

**Parameters:**
- `query` (str): Research query describing the medical imaging task

**Returns:**
- Dictionary containing:
  - `query`: The original query
  - `summary`: AI-generated comprehensive summary
  - `arxiv_results`: List of arXiv papers
  - `pubmed_results`: List of PubMed papers
  - `huggingface_results`: List of HuggingFace models
  - `total_papers`: Total number of papers found
  - `total_models`: Total number of models found

**Example:**
```python
results = agent.research("lung cancer detection")
print(f"Found {results['total_papers']} papers")
print(f"Summary: {results['summary']}")
```

#### `format_results(results: Dict) -> str`

Format research results for human-readable display.

**Parameters:**
- `results` (Dict): Results dictionary from `research()`

**Returns:**
- Formatted string with summary and sources

**Example:**
```python
results = agent.research("skin lesion classification")
formatted = agent.format_results(results)
print(formatted)
```

## MCP Tools

### ArxivTool

Tool for searching arXiv papers.

```python
from research_viz_agent.mcp_tools.arxiv_tool import ArxivTool

tool = ArxivTool()

# Search for papers
papers = tool.search_papers("medical imaging deep learning", max_results=10)

# Search specifically for medical CV papers
papers = tool.search_medical_cv_models("lung cancer")
```

#### Methods

- `search_papers(query, max_results=10, sort_by=Relevance)`: Generic paper search
- `search_medical_cv_models(additional_terms="")`: Medical CV-focused search

**Paper Result Structure:**
```python
{
    'title': str,
    'authors': List[str],
    'summary': str,
    'published': str,  # ISO format datetime
    'updated': str,    # ISO format datetime
    'categories': List[str],
    'pdf_url': str,
    'entry_id': str,
    'doi': str,
    'primary_category': str
}
```

### PubMedTool

Tool for searching PubMed papers.

```python
from research_viz_agent.mcp_tools.pubmed_tool import PubMedTool

tool = PubMedTool(email="your-email@example.com")

# Search for papers
papers = tool.search_papers("radiology AI", max_results=10)

# Search specifically for medical CV papers
papers = tool.search_medical_cv_models("pathology")
```

#### Methods

- `search_papers(query, max_results=10, sort="relevance")`: Generic paper search
- `search_medical_cv_models(additional_terms="")`: Medical CV-focused search

**Paper Result Structure:**
```python
{
    'pmid': str,
    'title': str,
    'authors': List[str],
    'abstract': str,
    'journal': str,
    'publication_date': str,
    'keywords': List[str],
    'mesh_terms': List[str],
    'doi': str
}
```

### HuggingFaceTool

Tool for searching HuggingFace models.

```python
from research_viz_agent.mcp_tools.huggingface_tool import HuggingFaceTool

tool = HuggingFaceTool(token="optional_hf_token")

# Search for models
models = tool.search_models("medical imaging", limit=20)

# Search specifically for medical CV models
models = tool.search_medical_cv_models("xray")

# Get detailed model info
info = tool.get_model_info("microsoft/resnet-50")
```

#### Methods

- `search_models(search_query, task=None, tags=None, limit=20)`: Generic model search
- `search_medical_cv_models(additional_query="")`: Medical CV-focused search
- `get_model_info(model_id)`: Get detailed info for a specific model

**Model Result Structure:**
```python
{
    'model_id': str,
    'author': str,
    'model_name': str,
    'downloads': int,
    'likes': int,
    'tags': List[str],
    'pipeline_tag': str,
    'created_at': str,
    'last_modified': str,
    'library_name': str,
    'model_card_url': str
}
```

## ResearchWorkflow

LangGraph workflow that coordinates the research process.

```python
from research_viz_agent.agents.research_workflow import ResearchWorkflow
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key="...", model_name="gpt-3.5-turbo")
workflow = ResearchWorkflow(llm, arxiv_tool, pubmed_tool, huggingface_tool)

# Run workflow
results = workflow.run("brain tumor segmentation")
```

### Workflow Steps

1. **search_arxiv**: Search arXiv for relevant papers
2. **search_pubmed**: Search PubMed for relevant papers
3. **search_huggingface**: Search HuggingFace for relevant models
4. **summarize**: Generate comprehensive summary using LLM

### State Structure

```python
{
    'query': str,
    'arxiv_results': List[Dict],
    'pubmed_results': List[Dict],
    'huggingface_results': List[Dict],
    'summary': str,
    'next_step': str
}
```

## Command-Line Interface

```bash
# Basic usage
python -m research_viz_agent.cli "lung cancer detection"

# With options
python -m research_viz_agent.cli "query" \
    --email your@email.com \
    --model gpt-4 \
    --temperature 0.5 \
    --output results.txt
```

### CLI Arguments

- `query` (required): Research query
- `--email`: Email for PubMed API (default: research@example.com)
- `--model`: OpenAI model name (default: gpt-3.5-turbo)
- `--temperature`: LLM temperature (default: 0.7)
- `--output`: Save results to file
