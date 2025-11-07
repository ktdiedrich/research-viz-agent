"""
Visualize RAG store embeddings and analyze clustering.

This script extracts embeddings from the ChromaDB vector store,
reduces dimensionality using UMAP/t-SNE, and creates visualizations
showing how documents cluster by source and semantic similarity.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Error: Missing required packages. Install with:")
    print("  uv pip install numpy matplotlib seaborn scikit-learn chromadb")
    sys.exit(1)

# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP not available. Install with: uv pip install umap-learn")


class RAGEmbeddingVisualizer:
    """Visualize and analyze RAG store embeddings."""
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """
        Initialize the visualizer.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
        """
        self.chroma_db_path = chroma_db_path
        self.client = None
        self.embeddings = None
        self.metadata = None
        self.documents = None
        self.ids = None
        
    def load_data(self) -> bool:
        """
        Load embeddings and metadata from ChromaDB.
        
        Returns:
            True if data loaded successfully
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get all collections
            collections = self.client.list_collections()
            
            if not collections:
                print(f"No collections found in {self.chroma_db_path}")
                return False
            
            print(f"Found {len(collections)} collection(s):")
            for col in collections:
                print(f"  - {col.name} ({col.count()} documents)")
            
            # Use the first collection (or let user specify)
            collection = collections[0]
            print(f"\nUsing collection: {collection.name}")
            
            # Get all data from collection
            results = collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
            
            # Check if we have embeddings
            embeddings_data = results.get('embeddings')
            if embeddings_data is None or len(embeddings_data) == 0:
                print("No embeddings found in collection")
                return False
            
            self.embeddings = np.array(embeddings_data)
            self.metadata = results['metadatas']
            self.documents = results['documents']
            self.ids = results['ids']
            
            print(f"Loaded {len(self.embeddings)} embeddings")
            print(f"Embedding dimension: {self.embeddings.shape[1]}")
            
            # Print source distribution
            sources = [m.get('source', 'unknown') for m in self.metadata]
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print("\nSource distribution:")
            for source, count in sorted(source_counts.items()):
                print(f"  {source}: {count}")
            
            return True
            
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return False
    
    def reduce_dimensions(self, method: str = 'tsne', n_components: int = 2) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            method: 'tsne' or 'umap'
            n_components: Number of dimensions (2 or 3)
            
        Returns:
            Reduced embeddings
        """
        print(f"\nReducing dimensions using {method.upper()}...")
        
        if method == 'umap':
            if not HAS_UMAP:
                print("UMAP not available, falling back to t-SNE")
                method = 'tsne'
            else:
                reducer = UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine'
                )
                return reducer.fit_transform(self.embeddings)
        
        if method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(self.embeddings) - 1),
                metric='cosine'
            )
            return reducer.fit_transform(self.embeddings)
        
        raise ValueError(f"Unknown method: {method}")
    
    def cluster_embeddings(self, method: str = 'kmeans', n_clusters: int = None) -> np.ndarray:
        """
        Cluster embeddings.
        
        Args:
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Cluster labels
        """
        print(f"\nClustering with {method.upper()}...")
        
        if method == 'kmeans':
            if n_clusters is None:
                # Estimate optimal number of clusters
                n_clusters = self._estimate_optimal_clusters()
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(self.embeddings)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:
                score = silhouette_score(self.embeddings, labels)
                print(f"Silhouette score: {score:.3f}")
            
            return labels
        
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            labels = clusterer.fit_predict(self.embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"Found {n_clusters} clusters")
            print(f"Noise points: {n_noise}")
            
            return labels
        
        raise ValueError(f"Unknown method: {method}")
    
    def _estimate_optimal_clusters(self, max_clusters: int = 10) -> int:
        """Estimate optimal number of clusters using elbow method."""
        if len(self.embeddings) < 10:
            return min(3, len(self.embeddings))
        
        max_k = min(max_clusters, len(self.embeddings) - 1)
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.embeddings)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection: find biggest drop
        diffs = np.diff(inertias)
        optimal_k = np.argmin(diffs) + 2
        
        print(f"Estimated optimal clusters: {optimal_k}")
        return optimal_k
    
    def plot_2d_scatter(
        self,
        embeddings_2d: np.ndarray,
        color_by: str = 'source',
        cluster_labels: Optional[np.ndarray] = None,
        output_file: Optional[str] = None
    ):
        """
        Create 2D scatter plot of embeddings.
        
        Args:
            embeddings_2d: 2D embeddings
            color_by: 'source', 'type', or 'cluster'
            cluster_labels: Cluster labels (if color_by='cluster')
            output_file: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Determine colors
        if color_by == 'source':
            sources = [m.get('source', 'unknown') for m in self.metadata]
            unique_sources = sorted(set(sources))
            color_map = {src: i for i, src in enumerate(unique_sources)}
            colors = [color_map[src] for src in sources]
            labels = sources
            title = "RAG Store Embeddings by Source"
            legend_title = "Source"
            
        elif color_by == 'type':
            types = [m.get('type', 'unknown') for m in self.metadata]
            unique_types = sorted(set(types))
            color_map = {t: i for i, t in enumerate(unique_types)}
            colors = [color_map[t] for t in types]
            labels = types
            title = "RAG Store Embeddings by Type"
            legend_title = "Type"
            
        elif color_by == 'cluster' and cluster_labels is not None:
            colors = cluster_labels
            labels = [f"Cluster {l}" if l >= 0 else "Noise" for l in cluster_labels]
            title = "RAG Store Embeddings - Clustered"
            legend_title = "Cluster"
            
        else:
            colors = ['blue'] * len(embeddings_2d)
            labels = ['Document'] * len(embeddings_2d)
            title = "RAG Store Embeddings"
            legend_title = None
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            cmap='tab10' if color_by in ['source', 'type'] else 'viridis',
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add labels for some points
        if len(embeddings_2d) <= 50:
            for i, (x, y) in enumerate(embeddings_2d):
                title_text = self.metadata[i].get('title', '')
                if title_text:
                    # Truncate long titles
                    short_title = title_text[:30] + '...' if len(title_text) > 30 else title_text
                    plt.annotate(
                        short_title,
                        (x, y),
                        fontsize=6,
                        alpha=0.7,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        
        # Add legend
        if legend_title and color_by in ['source', 'type']:
            unique_items = sorted(set(labels))
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                markersize=10, label=item)
                      for i, item in enumerate(unique_items)]
            plt.legend(handles=handles, title=legend_title, 
                      loc='best', framealpha=0.9)
        elif color_by == 'cluster':
            plt.colorbar(scatter, label='Cluster ID')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_file}")
        else:
            plt.show()
    
    def plot_3d_scatter(
        self,
        embeddings_3d: np.ndarray,
        color_by: str = 'source',
        output_file: Optional[str] = None
    ):
        """
        Create 3D scatter plot of embeddings.
        
        Args:
            embeddings_3d: 3D embeddings
            color_by: 'source' or 'type'
            output_file: Path to save plot
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine colors
        if color_by == 'source':
            sources = [m.get('source', 'unknown') for m in self.metadata]
            unique_sources = sorted(set(sources))
            color_map = {src: i for i, src in enumerate(unique_sources)}
            colors = [color_map[src] for src in sources]
            labels = unique_sources
            title = "RAG Store Embeddings - 3D View by Source"
            
        else:  # type
            types = [m.get('type', 'unknown') for m in self.metadata]
            unique_types = sorted(set(types))
            color_map = {t: i for i, t in enumerate(unique_types)}
            colors = [color_map[t] for t in types]
            labels = unique_types
            title = "RAG Store Embeddings - 3D View by Type"
        
        # Create scatter plot
        scatter = ax.scatter(
            embeddings_3d[:, 0],
            embeddings_3d[:, 1],
            embeddings_3d[:, 2],
            c=colors,
            cmap='tab10',
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_zlabel('Dimension 3', fontsize=12)
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab10(i / len(labels)), 
                            markersize=10, label=label)
                  for i, label in enumerate(labels)]
        ax.legend(handles=handles, title=color_by.capitalize(), 
                 loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_file}")
        else:
            plt.show()
    
    def analyze_clustering_quality(self, cluster_labels: np.ndarray) -> Dict:
        """
        Analyze clustering quality and characteristics.
        
        Args:
            cluster_labels: Cluster labels
            
        Returns:
            Dictionary with clustering metrics
        """
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        print("\n" + "=" * 60)
        print("CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Overall metrics
        if n_clusters > 1:
            silhouette = silhouette_score(self.embeddings, cluster_labels)
            print(f"\nSilhouette Score: {silhouette:.3f}")
            print(f"Number of Clusters: {n_clusters}")
        
        # Per-cluster analysis
        print("\nPer-Cluster Breakdown:")
        print("-" * 60)
        
        cluster_info = {}
        
        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                continue
                
            mask = cluster_labels == cluster_id
            cluster_docs = [self.metadata[i] for i, m in enumerate(mask) if m]
            
            # Source distribution
            sources = [d.get('source', 'unknown') for d in cluster_docs]
            source_counts = {}
            for src in sources:
                source_counts[src] = source_counts.get(src, 0) + 1
            
            # Type distribution
            types = [d.get('type', 'unknown') for d in cluster_docs]
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {sum(mask)} documents")
            print(f"  Sources: {source_counts}")
            print(f"  Types: {type_counts}")
            
            # Sample titles
            titles = [d.get('title', 'Untitled')[:60] for d in cluster_docs[:3]]
            if titles:
                print(f"  Sample titles:")
                for title in titles:
                    print(f"    - {title}")
            
            cluster_info[int(cluster_id)] = {
                'size': int(sum(mask)),
                'sources': source_counts,
                'types': type_counts,
                'sample_titles': titles
            }
        
        # Noise points
        if -1 in unique_clusters:
            n_noise = sum(cluster_labels == -1)
            print(f"\nNoise points: {n_noise}")
            cluster_info['noise'] = n_noise
        
        print("=" * 60)
        
        return cluster_info
    
    def create_html_report(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        output_file: str = "rag_embeddings_report.html"
    ):
        """
        Create interactive HTML report.
        
        Args:
            embeddings_2d: 2D embeddings
            cluster_labels: Optional cluster labels
            output_file: Path to save HTML file
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available. Install with: uv pip install plotly")
            return
        
        # Prepare data
        sources = [m.get('source', 'unknown') for m in self.metadata]
        types = [m.get('type', 'unknown') for m in self.metadata]
        titles = [m.get('title', 'Untitled') for m in self.metadata]
        urls = [m.get('url', '') for m in self.metadata]
        
        # Create DataFrame-like structure for plotly
        data = {
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'source': sources,
            'type': types,
            'title': titles,
            'url': urls
        }
        
        if cluster_labels is not None:
            data['cluster'] = [f"Cluster {l}" if l >= 0 else "Noise" 
                              for l in cluster_labels]
        
        # Create interactive scatter plot
        fig = px.scatter(
            data,
            x='x',
            y='y',
            color='source',
            hover_data=['title', 'type', 'url'],
            title='RAG Store Embeddings - Interactive View',
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            width=1200,
            height=800
        )
        
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=18),
            hovermode='closest'
        )
        
        # Save to HTML
        fig.write_html(output_file)
        print(f"\nInteractive report saved to {output_file}")


def ensure_output_dir(file_path: Optional[str]) -> None:
    """
    Create parent directory for the specified file path if needed.
    
    Args:
        file_path: Output file path. If None or has no directory, nothing is created.
    """
    if file_path:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RAG store embeddings and analyze clustering"
    )
    parser.add_argument(
        "--chroma-db",
        default="./chroma_db",
        help="Path to ChromaDB directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--method",
        choices=['tsne', 'umap'],
        default='tsne',
        help="Dimensionality reduction method (default: tsne)"
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Perform clustering analysis"
    )
    parser.add_argument(
        "--cluster-method",
        choices=['kmeans', 'dbscan'],
        default='kmeans',
        help="Clustering method (default: kmeans)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Number of clusters for k-means (auto-detect if not specified)"
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="three_d",
        help="Create 3D visualization"
    )
    parser.add_argument(
        "--output",
        help="Output file for plot (default: show plot)"
    )
    parser.add_argument(
        "--html",
        help="Create interactive HTML report"
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = RAGEmbeddingVisualizer(args.chroma_db)
    
    # Load data
    if not viz.load_data():
        return
    
    # Reduce dimensions
    n_components = 3 if args.three_d else 2
    embeddings_reduced = viz.reduce_dimensions(args.method, n_components)
    
    # Clustering
    cluster_labels = None
    if args.cluster:
        cluster_labels = viz.cluster_embeddings(
            args.cluster_method,
            args.n_clusters
        )
        viz.analyze_clustering_quality(cluster_labels)
    
    # Create visualizations
    if args.three_d:
        # Use user-specified output path as-is, or None to display
        output_file = args.output
        ensure_output_dir(output_file)
        viz.plot_3d_scatter(embeddings_reduced, color_by='source', output_file=output_file)
    else:
        # Create multiple plots
        if args.output:
            # Use user-specified output path as-is
            output_path = args.output
            ensure_output_dir(output_path)
            
            base, ext = os.path.splitext(output_path)
            
            # By source
            viz.plot_2d_scatter(embeddings_reduced, color_by='source', 
                              output_file=f"{base}_by_source{ext}")
            
            # By type
            viz.plot_2d_scatter(embeddings_reduced, color_by='type',
                              output_file=f"{base}_by_type{ext}")
            
            # By cluster
            if cluster_labels is not None:
                viz.plot_2d_scatter(embeddings_reduced, color_by='cluster',
                                  cluster_labels=cluster_labels,
                                  output_file=f"{base}_by_cluster{ext}")
        else:
            viz.plot_2d_scatter(embeddings_reduced, color_by='source')
            
            if cluster_labels is not None:
                viz.plot_2d_scatter(embeddings_reduced, color_by='cluster',
                                  cluster_labels=cluster_labels)
    
    # Create HTML report
    if args.html:
        # Use user-specified HTML path as-is
        html_path = args.html
        ensure_output_dir(html_path)
        viz.create_html_report(embeddings_reduced, cluster_labels, html_path)


if __name__ == "__main__":
    main()
