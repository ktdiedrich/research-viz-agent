"""
Demo script for RAG embedding visualization.

This script demonstrates how to visualize and analyze the embeddings
stored in the RAG database, showing clustering patterns and semantic
relationships between documents.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
import subprocess
import os


def populate_rag_store():
    """Populate RAG store with sample queries for demonstration."""
    print("=" * 80)
    print("STEP 1: Populating RAG Store with Sample Data")
    print("=" * 80)
    print()
    
    # Use GitHub provider (or none if no token)
    try:
        agent = MedicalCVResearchAgent(
            llm_provider="github" if os.getenv("GITHUB_TOKEN") else "none",
            pubmed_email="demo@example.com",
            enable_rag=True
        )
        
        # Run several queries to populate the RAG store
        queries = [
            "lung cancer detection",
            "brain tumor segmentation MRI",
            "skin lesion classification dermoscopy"
        ]
        
        for query in queries:
            print(f"\nResearching: {query}")
            print("-" * 60)
            
            try:
                results = agent.research(query)
                print(f"✓ Found {results.get('total_papers', 0)} papers, "
                      f"{results.get('total_models', 0)} models")
            except Exception as e:
                print(f"⚠ Error: {e}")
                continue
        
        # Show RAG stats
        print("\n" + "=" * 80)
        print("RAG Store Statistics:")
        print("=" * 80)
        stats = agent.get_rag_stats()
        print(f"Total documents: {stats.get('document_count', 0)}")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error populating RAG store: {e}")
        return False


def visualize_embeddings():
    """Visualize the embeddings using the visualization script."""
    print("=" * 80)
    print("STEP 2: Visualizing Embeddings")
    print("=" * 80)
    print()
    
    # Check which ChromaDB directories exist
    chroma_dirs = []
    for dirname in ["chroma_db", "chroma_db_github", "chroma_db_openai"]:
        if os.path.exists(dirname):
            chroma_dirs.append(dirname)
    
    if not chroma_dirs:
        print("No ChromaDB directories found. Please populate the RAG store first.")
        return False
    
    print(f"Found ChromaDB directories: {', '.join(chroma_dirs)}")
    chroma_db = chroma_dirs[0]
    print(f"Using: {chroma_db}\n")
    
    # Run visualization script
    script_path = Path(__file__).parent / "visualize_rag_embeddings.py"
    
    commands = [
        # 2D visualization with t-SNE
        [
            "python", str(script_path),
            "--chroma-db", chroma_db,
            "--method", "tsne",
            "--output", "rag_embeddings_2d.png"
        ],
        
        # With clustering
        [
            "python", str(script_path),
            "--chroma-db", chroma_db,
            "--method", "tsne",
            "--cluster",
            "--cluster-method", "kmeans",
            "--output", "rag_embeddings_clustered.png"
        ],
        
        # Interactive HTML report
        [
            "python", str(script_path),
            "--chroma-db", chroma_db,
            "--method", "tsne",
            "--cluster",
            "--html", "rag_embeddings_interactive.html"
        ]
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\nVisualization {i}/{len(commands)}:")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                print("✓ Success")
            else:
                print(f"⚠ Command exited with code {result.returncode}")
        except Exception as e:
            print(f"⚠ Error: {e}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - rag_embeddings_2d_by_source.png")
    print("  - rag_embeddings_2d_by_type.png")
    print("  - rag_embeddings_clustered_by_source.png")
    print("  - rag_embeddings_clustered_by_type.png")
    print("  - rag_embeddings_clustered_by_cluster.png")
    print("  - rag_embeddings_interactive.html (open in browser)")
    print()
    
    return True


def main():
    """Run the complete demo."""
    print("\n" + "=" * 80)
    print("RAG EMBEDDING VISUALIZATION DEMO")
    print("=" * 80)
    print()
    print("This demo will:")
    print("  1. Populate the RAG store with sample medical CV research")
    print("  2. Extract and visualize the embeddings")
    print("  3. Analyze clustering patterns")
    print()
    
    # Check if data already exists
    has_data = any(os.path.exists(d) for d in ["chroma_db", "chroma_db_github"])
    
    if has_data:
        response = input("RAG data already exists. Skip population step? (y/n): ")
        if response.lower() != 'y':
            populate_rag_store()
    else:
        populate_rag_store()
    
    # Visualize
    print("\nPress Enter to continue to visualization...")
    input()
    
    visualize_embeddings()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Open rag_embeddings_interactive.html in your browser")
    print("  - View the PNG files to see clustering patterns")
    print("  - Run your own queries and re-visualize to see changes")
    print()


if __name__ == "__main__":
    main()
