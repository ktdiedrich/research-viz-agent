#!/usr/bin/env python3
"""
Visualize RAG store additions.
"""
import argparse
from pathlib import Path
from research_viz_agent.utils.rag_tracker import (
    RAGTracker,
    create_bar_chart_ascii,
    create_bar_chart_html
)


def main():
    """Main function to visualize RAG store additions."""
    parser = argparse.ArgumentParser(
        description="Visualize RAG Store Query Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show ASCII bar chart
  %(prog)s

  # Show recent queries only
  %(prog)s --recent 5

  # Generate HTML chart
  %(prog)s --html rag_chart.html

  # Show summary statistics
  %(prog)s --summary

  # Clear tracking data
  %(prog)s --clear
        """
    )
    
    parser.add_argument(
        "--tracking-file",
        type=str,
        default="./chroma_db_github/rag_tracking.json",
        help="Path to tracking JSON file (default: ./chroma_db_github/rag_tracking.json)"
    )
    
    parser.add_argument(
        "--recent",
        type=int,
        help="Show only N most recent queries"
    )
    
    parser.add_argument(
        "--html",
        type=str,
        help="Generate HTML chart and save to file"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all tracking data"
    )
    
    args = parser.parse_args()
    
    # Check if tracking file exists
    tracking_path = Path(args.tracking_file)
    if not tracking_path.exists() and not args.clear:
        print(f"No tracking data found at: {args.tracking_file}")
        print(f"\nTracking data will be created automatically when you:")
        print(f"  1. Run research queries with RAG enabled")
        print(f"  2. Store results in the RAG database")
        print(f"\nExample:")
        print(f'  uv run python -m research_viz_agent.cli "lung cancer detection"')
        return
    
    # Initialize tracker
    tracker = RAGTracker(tracking_file=args.tracking_file)
    
    # Handle clear command
    if args.clear:
        confirm = input("Are you sure you want to clear all tracking data? (yes/no): ")
        if confirm.lower() == "yes":
            tracker.clear_tracking()
            print("✓ Tracking data cleared.")
        else:
            print("Cancelled.")
        return
    
    # Get queries
    if args.recent:
        queries = tracker.get_recent_queries(args.recent)
        print(f"\nShowing {len(queries)} most recent queries:")
    else:
        queries = tracker.get_all_queries()
    
    # Show summary if requested
    if args.summary:
        summary = tracker.get_summary()
        print("\n" + "=" * 60)
        print("RAG STORE TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Total Records: {summary['total_records']}")
        print(f"  - ArXiv: {summary['total_arxiv']}")
        print(f"  - PubMed: {summary['total_pubmed']}")
        print(f"  - HuggingFace: {summary['total_huggingface']}")
        print("=" * 60)
        return
    
    # Generate HTML chart if requested
    if args.html:
        create_bar_chart_html(queries, args.html)
        print(f"\n✓ HTML chart saved to: {args.html}")
        print(f"  Open it in your browser to view the visualization.")
        
        # Also show ASCII chart
        print(create_bar_chart_ascii(queries))
    else:
        # Show ASCII chart
        print(create_bar_chart_ascii(queries))
        
        # Show summary
        summary = tracker.get_summary()
        print(f"\nTotal: {summary['total_queries']} queries, {summary['total_records']} records")
        print(f"(ArXiv: {summary['total_arxiv']}, PubMed: {summary['total_pubmed']}, HuggingFace: {summary['total_huggingface']})")
        print(f"\nTracking file: {args.tracking_file}")
        print(f"\nTip: Use --html <file.html> to generate an interactive HTML chart")


if __name__ == "__main__":
    main()
