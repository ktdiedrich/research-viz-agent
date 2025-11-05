#!/usr/bin/env python3
"""
Demo script showing RAG tracking and visualization features.
"""
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
from research_viz_agent.utils.rag_tracker import create_bar_chart_ascii, create_bar_chart_html
import os


def main():
    """Demonstrate RAG tracking features."""
    print("=" * 80)
    print("RAG TRACKING DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize agent with RAG enabled
    print("1. Initializing agent with RAG tracking enabled...")
    agent = MedicalCVResearchAgent(
        llm_provider="github",
        enable_rag=True,
        rag_persist_dir="./demo_chroma_db"
    )
    
    # Run a few research queries
    queries = [
        "lung cancer detection",
        "brain tumor segmentation",
        "skin lesion classification"
    ]
    
    print("\n2. Running research queries and tracking additions...\n")
    for query in queries:
        print(f"   Researching: '{query}'")
        try:
            results = agent.research(query)
            print(f"   ✓ Found {results.get('total_papers', 0)} papers, "
                  f"{results.get('total_models', 0)} models")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        print()
    
    # Show tracking visualization
    print("\n3. Visualizing RAG store additions...\n")
    
    if agent.rag_store and agent.rag_store.tracker:
        tracker = agent.rag_store.tracker
        
        # Show ASCII chart
        queries_data = tracker.get_all_queries()
        print(create_bar_chart_ascii(queries_data))
        
        # Show summary
        summary = tracker.get_summary()
        print(f"\nSummary Statistics:")
        print(f"  Total Queries: {summary['total_queries']}")
        print(f"  Total Records: {summary['total_records']}")
        print(f"  - ArXiv: {summary['total_arxiv']}")
        print(f"  - PubMed: {summary['total_pubmed']}")
        print(f"  - HuggingFace: {summary['total_huggingface']}")
        
        # Generate HTML chart
        html_file = "./demo_rag_chart.html"
        create_bar_chart_html(queries_data, html_file)
        print(f"\n✓ HTML chart saved to: {html_file}")
        print(f"  Open it in your browser to see the interactive visualization!")
    else:
        print("RAG tracking not available (RAG might be disabled)")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nYou can now:")
    print("  1. Open demo_rag_chart.html in your browser")
    print("  2. Run: python -m research_viz_agent.cli --show-tracking")
    print("  3. Run: python -m research_viz_agent.cli --tracking-summary")
    print("  4. Run: python scripts/visualize_rag_tracking.py")


if __name__ == "__main__":
    main()
