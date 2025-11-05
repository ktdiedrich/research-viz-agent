# RAG Embedding Visualization - Quick Reference

## Installation

All dependencies are included in the project. Simply install the package:

```bash
uv pip install -e .
```

## Quick Commands

### Basic Visualization
```bash
# Simple 2D plot
uv run python scripts/visualize_rag_embeddings.py

# Save to file
uv run python scripts/visualize_rag_embeddings.py --output viz.png
```

### Clustering Analysis
```bash
# Auto-detect optimal clusters
uv run python scripts/visualize_rag_embeddings.py --cluster

# Specify number of clusters
uv run python scripts/visualize_rag_embeddings.py --cluster --n-clusters 5

# Use DBSCAN instead of k-means
uv run python scripts/visualize_rag_embeddings.py --cluster --cluster-method dbscan
```

### Advanced Options
```bash
# Use UMAP instead of t-SNE
uv run python scripts/visualize_rag_embeddings.py --method umap

# 3D visualization
uv run python scripts/visualize_rag_embeddings.py --3d

# Interactive HTML report
uv run python scripts/visualize_rag_embeddings.py --html report.html

# Different ChromaDB directory
uv run python scripts/visualize_rag_embeddings.py --chroma-db ./chroma_db_github
```

### Complete Analysis
```bash
# Generate all visualizations and report
uv run python scripts/visualize_rag_embeddings.py \
    --cluster \
    --output analysis.png \
    --html analysis_report.html
```

## Output Files

| File | Description |
|------|-------------|
| `*_by_source.png` | Color-coded by data source (ArXiv, PubMed, HuggingFace) |
| `*_by_type.png` | Color-coded by type (paper, model) |
| `*_by_cluster.png` | Color-coded by cluster assignment |
| `*.html` | Interactive report with hover tooltips |

## Understanding the Output

### Silhouette Score
- **> 0.7**: Strong clustering
- **0.5 - 0.7**: Reasonable structure
- **0.25 - 0.5**: Weak structure
- **< 0.25**: No clear clusters

### Visual Patterns
- **Tight clusters**: Related documents
- **Mixed colors**: Cross-source topics
- **Outliers**: Unique or off-topic documents
- **Gaps**: Semantic boundaries

## Common Use Cases

### 1. Understand Your RAG Store
```bash
uv run python scripts/visualize_rag_embeddings.py --html overview.html
```
Opens an interactive report showing all documents.

### 2. Find Research Themes
```bash
uv run python scripts/visualize_rag_embeddings.py --cluster
```
Identifies natural groupings and displays sample titles.

### 3. Check Coverage
```bash
uv run python scripts/visualize_rag_embeddings.py --output coverage.png
```
Visualize by source to see which databases cover which topics.

### 4. Quality Analysis
```bash
uv run python scripts/visualize_rag_embeddings.py --cluster --n-clusters 3
```
Test different cluster numbers to find optimal organization.

## Troubleshooting

### Error: No embeddings found
**Solution**: Ensure RAG store has data
```bash
uv run python -m research_viz_agent.cli "lung cancer detection"
```

### Error: Missing dependencies
**Solution**: Reinstall the package with all dependencies
```bash
uv pip install -e .
```

### Low silhouette score
**Solution**: This is normal for diverse queries. Try:
- More specific search terms
- Larger dataset (50+ documents)
- Different clustering methods

### Memory issues
**Solution**: For large datasets:
- Use UMAP instead of t-SNE: `--method umap`
- Reduce perplexity (edit script)
- Filter to specific time periods

## Tips

1. **Build corpus first**: Run 5-10 queries before visualizing
2. **Use focused queries**: Specific topics cluster better
3. **Try both methods**: t-SNE and UMAP show different patterns
4. **Check HTML report**: Hover over points to see details
5. **Compare source mixing**: Pure clusters may indicate gaps

## Example Workflow

```bash
# 1. Populate RAG store
uv run python -m research_viz_agent.cli "lung cancer detection"
uv run python -m research_viz_agent.cli "brain tumor MRI"
uv run python -m research_viz_agent.cli "skin lesion classification"

# 2. Analyze and visualize
uv run python scripts/visualize_rag_embeddings.py \
    --cluster \
    --output medical_cv.png \
    --html medical_cv_report.html

# 3. Open medical_cv_report.html in browser

# 4. Refine based on gaps found
```

## More Information

- Full documentation: [docs/RAG_ENCODING_CLUSTERING.md](RAG_ENCODING_CLUSTERING.md)
- Demo script: `examples/rag_embeddings_demo.py`
- Main README: [../README.md](../README.md)
