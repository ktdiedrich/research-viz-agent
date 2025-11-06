"""
Demo script showing CSV export functionality for research results and RAG searches.
"""
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
from research_viz_agent.utils.csv_export import export_rag_results_to_csv, export_research_results_to_csv


def demo_research_export():
    """Demonstrate exporting regular research results to CSV."""
    print("\n" + "="*60)
    print("DEMO 1: Export Regular Research Results to CSV")
    print("="*60)
    
    # Initialize agent without LLM (faster for demo)
    agent = MedicalCVResearchAgent(
        llm_provider="none",
        max_results=10,
        enable_rag=False  # Disable RAG for this demo
    )
    
    # Run research
    print("\nSearching for 'lung cancer detection'...")
    results = agent.research("lung cancer detection")
    
    print(f"Found {results['total_papers']} papers")
    
    # Export to CSV
    output_file = "research_export_demo.csv"
    export_research_results_to_csv(results, output_file)
    print(f"\n✓ Exported results to {output_file}")
    print(f"  Contains papers from ArXiv and PubMed")
    print(f"  Includes: title, authors, abstract, URL, publication date, etc.")


def demo_rag_export():
    """Demonstrate exporting RAG search results to CSV."""
    print("\n" + "="*60)
    print("DEMO 2: Export RAG Search Results to CSV")
    print("="*60)
    
    # Initialize agent with GitHub provider (free)
    agent = MedicalCVResearchAgent(
        llm_provider="github",
        max_results=5
    )
    
    # First, add some documents to RAG (if not already present)
    print("\nBuilding RAG database with sample queries...")
    agent.research("lung nodule detection")
    
    # Search RAG database
    print("\nSearching RAG database for 'deep learning imaging'...")
    rag_results = agent.search_rag(
        query="deep learning imaging",
        k=20
    )
    
    if rag_results['total_count'] > 0:
        print(f"Found {rag_results['total_count']} relevant documents")
        
        # Export to CSV
        output_file = "rag_search_demo.csv"
        export_rag_results_to_csv(rag_results, output_file)
        print(f"\n✓ Exported RAG search results to {output_file}")
        print(f"  Contains semantically relevant papers from your RAG database")
        print(f"  Includes: source, title, abstract, URL, authors, metadata")
    else:
        print("No results found in RAG database. Run more queries first.")


def demo_filtered_export():
    """Demonstrate exporting filtered RAG searches."""
    print("\n" + "="*60)
    print("DEMO 3: Export Filtered RAG Search (ArXiv only)")
    print("="*60)
    
    agent = MedicalCVResearchAgent(llm_provider="github")
    
    # Search with source filter
    print("\nSearching RAG database for ArXiv papers on 'CNN'...")
    rag_results = agent.search_rag(
        query="convolutional neural networks",
        k=15,
        source_filter="arxiv"
    )
    
    if rag_results['total_count'] > 0:
        print(f"Found {rag_results['total_count']} ArXiv papers")
        
        # Export to CSV
        output_file = "arxiv_cnn_demo.csv"
        export_rag_results_to_csv(rag_results, output_file)
        print(f"\n✓ Exported filtered results to {output_file}")
    else:
        print("No ArXiv papers found. Build RAG database first.")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("CSV EXPORT DEMOS - Medical CV Research Agent")
    print("="*60)
    
    try:
        # Demo 1: Regular research export
        demo_research_export()
        
        # Demo 2: RAG search export
        demo_rag_export()
        
        # Demo 3: Filtered export
        demo_filtered_export()
        
        print("\n" + "="*60)
        print("DEMOS COMPLETE")
        print("="*60)
        print("\nCSV files created:")
        print("  - research_export_demo.csv  (regular research)")
        print("  - rag_search_demo.csv       (RAG search)")
        print("  - arxiv_cnn_demo.csv        (filtered RAG search)")
        print("\nYou can open these in Excel, Google Sheets, or any CSV viewer.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have GITHUB_TOKEN or OPENAI_API_KEY set.")


if __name__ == "__main__":
    main()
