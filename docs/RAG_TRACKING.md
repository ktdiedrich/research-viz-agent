# RAG Store Query Tracking

This feature automatically tracks and visualizes queries and records added to the RAG (Retrieval-Augmented Generation) store.

## Overview

Every time you run a research query with RAG enabled, the system automatically tracks:
- The query text
- Number of ArXiv papers added
- Number of PubMed papers added
- Number of HuggingFace models added
- Timestamp of the query
- Embeddings provider used

## Tracking File Location

Tracking data is stored in a JSON file within your RAG directory:
- For GitHub provider: `./chroma_db_github/rag_tracking.json`
- For OpenAI provider: `./chroma_db/rag_tracking.json`
- Custom location: `<your_rag_dir>/rag_tracking.json`

## Usage

### Automatic Tracking

Tracking happens automatically when you run queries:

```bash
# Each query is automatically tracked
uv run python -m research_viz_agent.cli "lung cancer detection"
uv run python -m research_viz_agent.cli "brain tumor segmentation"
uv run python -m research_viz_agent.cli "skin lesion classification"
```

### View Tracking Chart

#### ASCII Bar Chart (Terminal)

```bash
# Show ASCII bar chart in terminal
uv run python -m research_viz_agent.cli --show-tracking

# Show summary statistics
uv run python -m research_viz_agent.cli --tracking-summary
```

#### HTML Interactive Chart

```bash
# Generate and view HTML chart
uv run python scripts/visualize_rag_tracking.py --html rag_chart.html

# Then open rag_chart.html in your browser
```

### Visualization Script

The `scripts/visualize_rag_tracking.py` script provides advanced visualization options:

```bash
# Show all queries with ASCII chart
uv run python scripts/visualize_rag_tracking.py

# Show only recent 5 queries
uv run python scripts/visualize_rag_tracking.py --recent 5

# Generate HTML chart
uv run python scripts/visualize_rag_tracking.py --html my_chart.html

# Show summary only
uv run python scripts/visualize_rag_tracking.py --summary

# Clear all tracking data
uv run python scripts/visualize_rag_tracking.py --clear

# Use custom tracking file location
uv run python scripts/visualize_rag_tracking.py --tracking-file ./custom_path/tracking.json
```

## Example Output

### ASCII Chart
```
================================================================================
RAG STORE ADDITIONS - BAR CHART
================================================================================

 1. lung cancer detection
    2025-01-15 10:30
    ████████████████████████████████ 45 total
    (ArXiv: 20, PubMed: 20, HuggingFace: 5)

 2. brain tumor segmentation
    2025-01-15 11:45
    ████████████████████████████ 38 total
    (ArXiv: 18, PubMed: 15, HuggingFace: 5)

 3. skin lesion classification
    2025-01-15 14:20
    ███████████████████████████████████ 52 total
    (ArXiv: 22, PubMed: 25, HuggingFace: 5)

================================================================================

Total: 3 queries, 135 records
(ArXiv: 60, PubMed: 60, HuggingFace: 15)
```

### HTML Chart

The HTML chart provides:
- Color-coded bars for each query
- Source breakdown (ArXiv/PubMed/HuggingFace)
- Timestamps
- Summary statistics
- Responsive design

## Python API

You can also access tracking programmatically:

```python
from research_viz_agent.utils.rag_tracker import RAGTracker, create_bar_chart_ascii, create_bar_chart_html

# Initialize tracker
tracker = RAGTracker(tracking_file="./chroma_db_github/rag_tracking.json")

# Get all queries
queries = tracker.get_all_queries()

# Get recent queries
recent = tracker.get_recent_queries(limit=5)

# Get summary statistics
summary = tracker.get_summary()
print(f"Total queries: {summary['total_queries']}")
print(f"Total records: {summary['total_records']}")

# Create ASCII chart
ascii_chart = create_bar_chart_ascii(queries)
print(ascii_chart)

# Create HTML chart
create_bar_chart_html(queries, "my_chart.html")

# Clear tracking
tracker.clear_tracking()
```

## Demo

Run the included demo to see tracking in action:

```bash
uv run python examples/rag_tracking_demo.py
```

This will:
1. Run several research queries
2. Show tracking data
3. Generate an HTML chart
4. Display summary statistics

## Benefits

- **Track Research History**: See what queries you've run and when
- **Monitor Growth**: Visualize how your RAG database grows over time
- **Source Distribution**: Understand which sources contribute most records
- **Performance Insights**: Identify which queries return the most results
- **Quality Control**: Verify that results are being stored correctly

## Technical Details

### Tracking Data Structure

The tracking JSON file contains:
```json
{
  "queries": [
    {
      "timestamp": "2025-01-15T10:30:00",
      "query": "lung cancer detection",
      "arxiv_count": 20,
      "pubmed_count": 20,
      "huggingface_count": 5,
      "total_added": 45,
      "embeddings_provider": "github"
    }
  ],
  "total_records": 45
}
```

### Implementation

- `RAGTracker`: Core tracking class (in `utils/rag_tracker.py`)
- `ResearchRAGStore`: Automatically calls tracker when storing results
- `create_bar_chart_ascii()`: Generates terminal-friendly charts
- `create_bar_chart_html()`: Generates interactive HTML charts

## Troubleshooting

**No tracking file found:**
- Tracking file is created automatically on first use
- Make sure RAG is enabled (don't use `--no-rag`)
- Run at least one query first

**Tracking not updating:**
- Verify RAG is enabled
- Check file permissions
- Ensure you're looking at the correct provider's directory

**Want to reset tracking:**
```bash
uv run python scripts/visualize_rag_tracking.py --clear
```
