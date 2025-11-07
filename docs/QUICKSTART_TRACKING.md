# Quick Start Guide: RAG Query Tracking

## Overview

Every time you run a research query, the system **automatically tracks** what was added to your RAG database and provides **beautiful visualizations**.

## Step 1: Run Some Queries

```bash
# Run your research queries as normal
uv run python -m research_viz_agent.cli "lung cancer detection"
uv run python -m research_viz_agent.cli "brain tumor segmentation"  
uv run python -m research_viz_agent.cli "skin lesion classification"
```

Behind the scenes, each query is automatically tracked! ✨

## Step 2: View the Tracking Chart

```bash
# See what's been added in a nice bar chart
uv run python -m research_viz_agent.cli --show-tracking
```

**Output:**
```
================================================================================
RAG STORE ADDITIONS - BAR CHART
================================================================================

 1. lung cancer detection
    2025-11-05 10:30
    █████████████████████████████████ 43 total
    (ArXiv: 20, PubMed: 18, HuggingFace: 5)

 2. brain tumor segmentation
    2025-11-05 11:45
    ████████████████████████ 30 total
    (ArXiv: 15, PubMed: 12, HuggingFace: 3)

 3. skin lesion classification
    2025-11-05 14:20
    ████████████████████████████████████ 48 total
    (ArXiv: 22, PubMed: 20, HuggingFace: 6)

================================================================================
```

## Step 3: Get Summary Statistics

```bash
uv run python -m research_viz_agent.cli --tracking-summary
```

**Output:**
```
============================================================
RAG STORE TRACKING SUMMARY
============================================================
Total Queries: 3
Total Records: 121
  - ArXiv: 57
  - PubMed: 50
  - HuggingFace: 14
============================================================
```

## Step 4: Generate HTML Chart (Optional)

```bash
# Create a beautiful HTML chart
uv run python scripts/visualize_rag_tracking.py --html my_research_chart.html

# Open in browser
xdg-open my_research_chart.html  # Linux
open my_research_chart.html      # macOS
start my_research_chart.html     # Windows
```

The HTML chart includes:
- 📊 Color-coded bars for each query
- 🕐 Timestamps
- 📈 Source breakdown (ArXiv/PubMed/HuggingFace)
- 💫 Professional styling
- 📱 Responsive design

## Advanced Options

```bash
# Show only recent 5 queries
uv run python scripts/visualize_rag_tracking.py --recent 5

# View summary only
uv run python scripts/visualize_rag_tracking.py --summary

# Clear all tracking data
uv run python scripts/visualize_rag_tracking.py --clear
```

## Where is the Data Stored?

Tracking data is automatically saved in:
- **GitHub provider**: `./chroma_db_github/rag_tracking.json`
- **OpenAI provider**: `./chroma_db/rag_tracking.json`

This file persists across sessions and grows with each query!

## Key Benefits

✅ **Zero Configuration** - Works automatically  
✅ **Visual Insights** - See what's in your database at a glance  
✅ **Quality Control** - Verify queries are returning results  
✅ **Research History** - Track all your queries over time  
✅ **Source Analysis** - Understand which sources contribute most  

## Example Workflow

```bash
# Monday: Research lung diseases
uv run python -m research_viz_agent.cli "lung cancer detection"
uv run python -m research_viz_agent.cli "COPD imaging"

# Tuesday: Research brain conditions  
uv run python -m research_viz_agent.cli "brain tumor segmentation"
uv run python -m research_viz_agent.cli "Alzheimer's MRI"

# End of week: Review what you've accumulated
uv run python -m research_viz_agent.cli --show-tracking

# Generate report for your team
uv run python scripts/visualize_rag_tracking.py --html weekly_research.html
```

## Questions?

- **Tracking not showing?** Make sure RAG is enabled (don't use `--no-rag`)
- **Want to start fresh?** Use `--clear` to reset tracking
- **Need more details?** See [docs/RAG_TRACKING.md](docs/RAG_TRACKING.md)

---

**That's it!** Your research queries are now automatically tracked and beautifully visualized. Happy researching! 🎉
