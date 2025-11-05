# RAG Store Encoding and Clustering Analysis

## Overview

The RAG (Retrieval-Augmented Generation) store uses vector embeddings to encode semantic meaning of research documents. This document explains how documents are encoded, stored, and how clustering analysis reveals semantic relationships.

## How Documents Are Encoded

### 1. Text to Embeddings

When a document (paper or model) is added to the RAG store:

1. **Text Extraction**: Key information is extracted and formatted:
   - **Papers**: Title, authors, abstract, categories, keywords
   - **Models**: Model ID, description, tags, task type

2. **Embedding Generation**: The text is converted to a high-dimensional vector using:
   - **GitHub Models**: `text-embedding-3-small` (1536 dimensions)
   - **OpenAI**: `text-embedding-3-small` or `text-embedding-ada-002` (1536 dimensions)

3. **Vector Storage**: The embedding vector is stored in ChromaDB alongside metadata

### 2. Semantic Similarity

Embeddings capture semantic meaning:
- Similar documents have vectors close together in high-dimensional space
- Distance metrics (cosine similarity) measure semantic similarity
- Queries are also embedded and matched to stored documents

## ChromaDB Storage Structure

```
chroma_db_github/  (or chroma_db/ for OpenAI)
├── chroma.sqlite3         # Metadata database
└── [UUID]/                # Collection data
    ├── data_level0.bin    # Vector data
    ├── header.bin         # Index header
    └── link_lists.bin     # HNSW graph links
```

### Metadata Stored

Each document includes:
- **source**: `arxiv`, `pubmed`, or `huggingface`
- **type**: `paper` or `model`
- **title**: Document title
- **authors**: Author list (papers)
- **url**: Link to original source
- **query**: Original search query
- **indexed_at**: Timestamp
- **[source-specific fields]**: PMIDs, entry IDs, tags, etc.

## Clustering Analysis

### Purpose

Clustering reveals:
- **Semantic groupings**: Which documents are conceptually similar
- **Research themes**: Common topics across different sources
- **Quality metrics**: How well-separated the clusters are

### Algorithms Used

#### K-Means Clustering
- Partitions documents into K clusters
- Minimizes within-cluster variance
- Optimal K estimated using elbow method
- Best for: Roughly spherical, evenly-sized clusters

#### DBSCAN Clustering
- Density-based clustering
- Finds clusters of arbitrary shape
- Automatically detects outliers (noise points)
- Best for: Irregular shapes, varying densities

### Quality Metrics

**Silhouette Score** (range: -1 to 1):
- **>0.7**: Strong, well-separated clusters
- **0.5-0.7**: Reasonable structure
- **0.25-0.5**: Weak structure
- **<0.25**: No meaningful structure

The score measures:
- How similar a document is to its own cluster
- vs. how similar it is to other clusters

## Visualization Methods

### Dimensionality Reduction

High-dimensional embeddings (1536D) are reduced to 2D/3D for visualization:

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Preserves local structure (nearby points stay nearby)
- Good for revealing clusters
- Non-linear transformation
- Parameters:
  - `perplexity`: Local neighborhood size (default: 30)
  - `metric`: Distance measure (cosine for embeddings)

#### UMAP (Uniform Manifold Approximation and Projection)
- Preserves both local and global structure
- Faster than t-SNE for large datasets
- More stable across runs
- Parameters:
  - `n_neighbors`: Local neighborhood size (default: 15)
  - `min_dist`: Minimum distance between points (default: 0.1)

### Color Coding

Visualizations use different color schemes:
- **By Source**: ArXiv (blue), PubMed (orange), HuggingFace (green)
- **By Type**: Paper (blue), Model (orange)
- **By Cluster**: Gradient showing cluster assignments

## Interpreting Results

### What Clusters Mean

**Tight Clusters**: Documents in the same cluster are semantically related:
- Similar research topics
- Related medical conditions
- Common AI techniques

**Mixed-Source Clusters**: Papers from different sources on the same topic:
- Shows cross-database semantic similarity
- Indicates comprehensive topic coverage

**Source-Specific Clusters**: Clusters dominated by one source:
- May indicate unique terminology
- Different focus areas per database
- Technical vs. clinical language differences

### Example Analysis

From the test visualization:

```
Cluster 0: 25 documents
  Sources: {'arxiv': 25}
  Types: {'paper': 25}
  Themes: Radiology, chest X-ray, report generation
  Interpretation: Pure ArXiv cluster focused on medical imaging AI

Cluster 1: 55 documents  
  Sources: {'arxiv': 17, 'pubmed': 38}
  Types: {'paper': 55}
  Themes: Various medical AI applications
  Interpretation: Mixed-source cluster with broader medical AI focus
```

**Silhouette Score: 0.069**
- Indicates weak clustering structure
- Documents are semantically diverse
- May benefit from more specific queries or more clusters

**Quick Start**:
- All dependencies are included with the package installation: `uv pip install -e .`

### Output Files

When using `--output viz.png`, multiple files are created:
- `viz_by_source.png`: Color-coded by source (ArXiv, PubMed, HuggingFace)
- `viz_by_type.png`: Color-coded by type (paper, model)
- `viz_by_cluster.png`: Color-coded by cluster assignment (if --cluster used)

## Best Practices

### For Better Clustering

1. **Focused Queries**: Use specific medical CV topics
   - Good: "lung cancer detection CT scans"
   - Poor: "medical imaging"

2. **Sufficient Data**: Aim for 50+ documents for meaningful clusters
   - Run multiple related queries
   - Build up the RAG store over time

3. **Consistent Embeddings**: Use the same provider (GitHub/OpenAI)
   - Different providers create incompatible embeddings
   - Each provider has its own ChromaDB directory

### Interpreting Visualizations

1. **Look for Gaps**: Empty spaces indicate semantic boundaries
2. **Check Overlap**: Mixed-source clusters show comprehensive coverage
3. **Identify Outliers**: Lone points may be off-topic or unique
4. **Compare Cluster Sizes**: Imbalanced clusters may need refinement

## Technical Details

### Embedding Model Specifications

**GitHub Models (text-embedding-3-small)**:
- Dimensions: 1536
- Max tokens: 8191
- Provider: Azure OpenAI via GitHub
- Cost: Free with GitHub Pro

**OpenAI (text-embedding-3-small)**:
- Dimensions: 1536
- Max tokens: 8191  
- Provider: OpenAI directly
- Cost: $0.02 per 1M tokens

### Performance Considerations

**Memory**: Each embedding is ~6KB (1536 × 4 bytes)
- 1000 documents ≈ 6 MB
- 10000 documents ≈ 60 MB

**Speed**: 
- Embedding generation: ~0.1s per document
- t-SNE reduction: ~1-10s depending on dataset size
- UMAP reduction: ~0.5-5s (faster than t-SNE)
- Clustering: <1s for <10000 documents

## Troubleshooting

### No Embeddings Found
- Check ChromaDB path is correct
- Ensure documents have been added to RAG store
- Verify collection name matches

### Poor Clustering Quality (Low Silhouette Score)
- Normal for diverse queries
- Try more specific search terms
- Increase number of clusters
- Use DBSCAN to detect natural groupings

### Memory Issues
- Reduce dataset size with query filtering
- Use smaller `perplexity` for t-SNE
- Use UMAP instead of t-SNE

### Visualization Issues
- Ensure all dependencies are installed: `uv pip install -e .`
- Verify the ChromaDB path is correct
- Check that the RAG store has data (run some queries first)

## Future Enhancements

Potential improvements:
- **Topic modeling**: Extract themes from clusters
- **Time series**: Track how embeddings evolve
- **Query optimization**: Suggest related queries
- **Similarity search**: Find documents similar to a given one
- **Duplicate detection**: Identify near-duplicate papers

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [t-SNE Paper](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [UMAP Paper](https://arxiv.org/abs/1802.03426)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
