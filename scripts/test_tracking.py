#!/usr/bin/env python3
"""
Quick test of RAG tracking functionality.
"""
import tempfile
import os
from research_viz_agent.utils.rag_tracker import (
    RAGTracker,
    create_bar_chart_ascii,
    create_bar_chart_html
)


def test_tracking():
    """Test basic tracking functionality."""
    print("Testing RAG Tracker...")
    
    # Create temporary tracking file
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_file = os.path.join(tmpdir, "test_tracking.json")
        
        # Initialize tracker
        tracker = RAGTracker(tracking_file=tracking_file)
        print("✓ Tracker initialized")
        
        # Track some queries
        tracker.track_query(
            query="lung cancer detection",
            arxiv_count=20,
            pubmed_count=15,
            huggingface_count=5,
            embeddings_provider="github"
        )
        print("✓ First query tracked")
        
        tracker.track_query(
            query="brain tumor segmentation",
            arxiv_count=18,
            pubmed_count=12,
            huggingface_count=3,
            embeddings_provider="github"
        )
        print("✓ Second query tracked")
        
        # Get summary
        summary = tracker.get_summary()
        print(f"\n✓ Summary: {summary['total_queries']} queries, {summary['total_records']} records")
        
        assert summary['total_queries'] == 2
        assert summary['total_records'] == 73
        assert summary['total_arxiv'] == 38
        assert summary['total_pubmed'] == 27
        assert summary['total_huggingface'] == 8
        
        # Get queries
        queries = tracker.get_all_queries()
        assert len(queries) == 2
        print(f"✓ Retrieved {len(queries)} queries")
        
        # Test ASCII chart
        ascii_chart = create_bar_chart_ascii(queries)
        assert "lung cancer detection" in ascii_chart
        assert "brain tumor segmentation" in ascii_chart
        print("✓ ASCII chart created")
        
        # Test HTML chart
        html_file = os.path.join(tmpdir, "test_chart.html")
        create_bar_chart_html(queries, html_file)
        assert os.path.exists(html_file)
        with open(html_file, 'r') as f:
            html_content = f.read()
            assert "lung cancer detection" in html_content
        print("✓ HTML chart created")
        
        # Test recent queries
        recent = tracker.get_recent_queries(limit=1)
        assert len(recent) == 1
        assert recent[0]['query'] == "brain tumor segmentation"
        print("✓ Recent queries working")
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_tracking()
