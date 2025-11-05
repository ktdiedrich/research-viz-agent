# RAG Store Query Tracking Feature - Summary

## What Was Added

A comprehensive tracking and visualization system for monitoring queries and records added to the RAG (Retrieval-Augmented Generation) store.

## New Files Created

1. **`research_viz_agent/utils/rag_tracker.py`**
   - Core tracking module
   - `RAGTracker` class for managing tracking data
   - `create_bar_chart_ascii()` - Terminal-friendly bar charts
   - `create_bar_chart_html()` - Interactive HTML charts

2. **`scripts/visualize_rag_tracking.py`**
   - Standalone script for visualizing tracking data
   - Supports ASCII and HTML output
   - Options for filtering, summary stats, and clearing data

3. **`examples/rag_tracking_demo.py`**
   - Demonstration script showing tracking features
   - Runs sample queries and generates visualizations

4. **`docs/RAG_TRACKING.md`**
   - Complete documentation for the tracking feature
   - Usage examples, API reference, troubleshooting

5. **`scripts/test_tracking.py`**
   - Test script verifying tracking functionality

## Modified Files

1. **`research_viz_agent/utils/rag_store.py`**
   - Added `RAGTracker` import and integration
   - Modified `__init__()` to initialize tracker
   - Updated `store_research_results()` to track additions
   - Now returns count of added records

2. **`research_viz_agent/cli.py`**
   - Added `--show-tracking` flag for viewing charts
   - Added `--tracking-summary` flag for statistics
   - Integrated tracking visualization commands

3. **`README.md`**
   - Added tracking feature to Features list
   - Added RAG Query Tracking & Visualization section
   - Included usage examples

## How It Works

### Automatic Tracking

Every time `store_research_results()` is called (which happens automatically during research):
1. Counts records from each source (ArXiv, PubMed, HuggingFace)
2. Records query text, counts, timestamp, and provider
3. Saves to `rag_tracking.json` in the RAG directory
4. Cumulative totals are updated

### Data Storage

Tracking data is stored in JSON format:
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

### Visualization

**ASCII Charts (Terminal):**
```
================================================================================
RAG STORE ADDITIONS - BAR CHART
================================================================================

 1. lung cancer detection
    2025-01-15 10:30
    ████████████████████████████████ 45 total
    (ArXiv: 20, PubMed: 20, HuggingFace: 5)
================================================================================
```

**HTML Charts (Browser):**
- Color-coded bars
- Interactive layout
- Source breakdowns
- Summary statistics
- Professional styling

## Usage Examples

### CLI Commands

```bash
# View tracking in terminal
uv run python -m research_viz_agent.cli --show-tracking

# Show summary only
uv run python -m research_viz_agent.cli --tracking-summary

# Generate HTML chart
uv run python scripts/visualize_rag_tracking.py --html chart.html

# Show recent 5 queries
uv run python scripts/visualize_rag_tracking.py --recent 5
```

### Python API

```python
from research_viz_agent.utils.rag_tracker import RAGTracker

tracker = RAGTracker("./chroma_db_github/rag_tracking.json")

# Get summary
summary = tracker.get_summary()
print(f"Total: {summary['total_records']} records")

# Get all queries
queries = tracker.get_all_queries()

# Generate chart
from research_viz_agent.utils.rag_tracker import create_bar_chart_ascii
print(create_bar_chart_ascii(queries))
```

## Benefits

1. **Transparency**: See exactly what was added to your RAG store
2. **Monitoring**: Track database growth over time
3. **Quality Control**: Verify queries are producing results
4. **Analytics**: Understand which sources contribute most
5. **History**: Review past research queries
6. **Visualization**: Clear, intuitive bar charts

## Technical Details

- **Provider-Specific**: Tracks separately for GitHub and OpenAI providers
- **Automatic**: No manual tracking needed
- **Persistent**: Survives across sessions
- **Lightweight**: Minimal overhead (JSON file)
- **Non-Breaking**: Existing code continues to work
- **Backward Compatible**: Works with existing RAG stores

## Testing

Run the test script to verify functionality:
```bash
uv run python scripts/test_tracking.py
```

All tests pass successfully ✅

## Next Steps

Users can now:
1. Run queries as usual - tracking happens automatically
2. View tracking charts with `--show-tracking`
3. Generate HTML reports for presentations
4. Monitor RAG database growth
5. Analyze which queries are most productive
