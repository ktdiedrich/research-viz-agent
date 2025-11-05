"""
Tests for RAG tracker functionality.
"""
import json
import os

from research_viz_agent.utils.rag_tracker import (
    RAGTracker,
    create_bar_chart_ascii,
    create_bar_chart_html
)


class TestRAGTrackerInitialization:
    """Test RAGTracker initialization."""
    
    def test_init_creates_new_tracker(self, tmp_path):
        """Test initializing a new tracker."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        assert tracker.tracking_file == str(tracking_file)
        assert tracker.data == {"queries": [], "total_records": 0}
    
    def test_init_loads_existing_data(self, tmp_path):
        """Test loading existing tracking data."""
        tracking_file = tmp_path / "tracking.json"
        
        # Create existing data
        existing_data = {
            "queries": [
                {
                    "timestamp": "2025-01-01T12:00:00",
                    "query": "test query",
                    "arxiv_count": 5,
                    "pubmed_count": 3,
                    "huggingface_count": 2,
                    "total_added": 10,
                    "embeddings_provider": "openai"
                }
            ],
            "total_records": 10
        }
        
        with open(tracking_file, 'w') as f:
            json.dump(existing_data, f)
        
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        assert len(tracker.data["queries"]) == 1
        assert tracker.data["total_records"] == 10
        assert tracker.data["queries"][0]["query"] == "test query"
    
    def test_init_handles_corrupted_json(self, tmp_path):
        """Test handling corrupted JSON file."""
        tracking_file = tmp_path / "tracking.json"
        
        # Write invalid JSON
        with open(tracking_file, 'w') as f:
            f.write("{ invalid json }")
        
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        # Should initialize with empty data
        assert tracker.data == {"queries": [], "total_records": 0}
    
    def test_init_with_default_path(self):
        """Test initialization with default path."""
        tracker = RAGTracker()
        
        assert tracker.tracking_file == "./rag_tracking.json"
        assert tracker.data == {"queries": [], "total_records": 0}


class TestRAGTrackerTrackQuery:
    """Test tracking queries."""
    
    def test_track_query_basic(self, tmp_path):
        """Test tracking a basic query."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query(
            query="test query",
            arxiv_count=5,
            pubmed_count=3,
            huggingface_count=2,
            embeddings_provider="openai"
        )
        
        assert len(tracker.data["queries"]) == 1
        assert tracker.data["total_records"] == 10
        
        query_entry = tracker.data["queries"][0]
        assert query_entry["query"] == "test query"
        assert query_entry["arxiv_count"] == 5
        assert query_entry["pubmed_count"] == 3
        assert query_entry["huggingface_count"] == 2
        assert query_entry["total_added"] == 10
        assert query_entry["embeddings_provider"] == "openai"
        assert "timestamp" in query_entry
    
    def test_track_query_saves_to_file(self, tmp_path):
        """Test that tracking saves to file."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query(
            query="test query",
            arxiv_count=1,
            pubmed_count=2,
            huggingface_count=3
        )
        
        # Verify file was created
        assert tracking_file.exists()
        
        # Load and verify data
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["queries"]) == 1
        assert data["total_records"] == 6
    
    def test_track_multiple_queries(self, tmp_path):
        """Test tracking multiple queries."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=5)
        tracker.track_query("query 2", pubmed_count=3)
        tracker.track_query("query 3", huggingface_count=2)
        
        assert len(tracker.data["queries"]) == 3
        assert tracker.data["total_records"] == 10
        assert tracker.data["queries"][0]["query"] == "query 1"
        assert tracker.data["queries"][1]["query"] == "query 2"
        assert tracker.data["queries"][2]["query"] == "query 3"
    
    def test_track_query_with_zeros(self, tmp_path):
        """Test tracking query with zero counts."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query(
            query="empty query",
            arxiv_count=0,
            pubmed_count=0,
            huggingface_count=0
        )
        
        assert len(tracker.data["queries"]) == 1
        assert tracker.data["total_records"] == 0
        assert tracker.data["queries"][0]["total_added"] == 0
    
    def test_track_query_creates_directory(self, tmp_path):
        """Test that tracking creates necessary directories."""
        tracking_file = tmp_path / "subdir" / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("test", arxiv_count=1)
        
        assert tracking_file.exists()
        assert tracking_file.parent.exists()


class TestRAGTrackerGetMethods:
    """Test getter methods."""
    
    def test_get_all_queries_empty(self, tmp_path):
        """Test getting all queries when empty."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        queries = tracker.get_all_queries()
        
        assert queries == []
    
    def test_get_all_queries_with_data(self, tmp_path):
        """Test getting all queries with data."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=1)
        tracker.track_query("query 2", pubmed_count=2)
        
        queries = tracker.get_all_queries()
        
        assert len(queries) == 2
        assert queries[0]["query"] == "query 1"
        assert queries[1]["query"] == "query 2"
    
    def test_get_total_records_empty(self, tmp_path):
        """Test getting total records when empty."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        total = tracker.get_total_records()
        
        assert total == 0
    
    def test_get_total_records_with_data(self, tmp_path):
        """Test getting total records with data."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=5)
        tracker.track_query("query 2", pubmed_count=3, huggingface_count=2)
        
        total = tracker.get_total_records()
        
        assert total == 10
    
    def test_get_recent_queries_less_than_limit(self, tmp_path):
        """Test getting recent queries with fewer than limit."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=1)
        tracker.track_query("query 2", arxiv_count=1)
        
        recent = tracker.get_recent_queries(limit=10)
        
        assert len(recent) == 2
    
    def test_get_recent_queries_more_than_limit(self, tmp_path):
        """Test getting recent queries with more than limit."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        # Add 15 queries
        for i in range(15):
            tracker.track_query(f"query {i}", arxiv_count=1)
        
        recent = tracker.get_recent_queries(limit=5)
        
        assert len(recent) == 5
        # Should get the last 5
        assert recent[0]["query"] == "query 10"
        assert recent[-1]["query"] == "query 14"
    
    def test_get_recent_queries_empty(self, tmp_path):
        """Test getting recent queries when empty."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        recent = tracker.get_recent_queries()
        
        assert recent == []


class TestRAGTrackerGetSummary:
    """Test summary statistics."""
    
    def test_get_summary_empty(self, tmp_path):
        """Test getting summary when empty."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        summary = tracker.get_summary()
        
        assert summary == {
            "total_queries": 0,
            "total_records": 0,
            "total_arxiv": 0,
            "total_pubmed": 0,
            "total_huggingface": 0
        }
    
    def test_get_summary_with_data(self, tmp_path):
        """Test getting summary with data."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=5, pubmed_count=3)
        tracker.track_query("query 2", pubmed_count=2, huggingface_count=4)
        tracker.track_query("query 3", arxiv_count=1, huggingface_count=1)
        
        summary = tracker.get_summary()
        
        assert summary["total_queries"] == 3
        assert summary["total_records"] == 16
        assert summary["total_arxiv"] == 6
        assert summary["total_pubmed"] == 5
        assert summary["total_huggingface"] == 5
    
    def test_get_summary_single_source(self, tmp_path):
        """Test summary with only one source."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=10)
        
        summary = tracker.get_summary()
        
        assert summary["total_queries"] == 1
        assert summary["total_records"] == 10
        assert summary["total_arxiv"] == 10
        assert summary["total_pubmed"] == 0
        assert summary["total_huggingface"] == 0


class TestRAGTrackerClearTracking:
    """Test clearing tracking data."""
    
    def test_clear_tracking_empty(self, tmp_path):
        """Test clearing already empty tracker."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.clear_tracking()
        
        assert tracker.data == {"queries": [], "total_records": 0}
    
    def test_clear_tracking_with_data(self, tmp_path):
        """Test clearing tracker with data."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        tracker.track_query("query 1", arxiv_count=5)
        tracker.track_query("query 2", pubmed_count=3)
        
        assert len(tracker.data["queries"]) == 2
        assert tracker.data["total_records"] == 8
        
        tracker.clear_tracking()
        
        assert tracker.data == {"queries": [], "total_records": 0}
        
        # Verify file was updated
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        assert data == {"queries": [], "total_records": 0}


class TestCreateBarChartAscii:
    """Test ASCII bar chart creation."""
    
    def test_create_bar_chart_empty(self):
        """Test creating chart with no queries."""
        result = create_bar_chart_ascii([])
        
        assert result == "No queries tracked yet."
    
    def test_create_bar_chart_zero_records(self):
        """Test creating chart with queries but no records."""
        queries = [
            {
                "query": "test",
                "total_added": 0,
                "arxiv_count": 0,
                "pubmed_count": 0,
                "huggingface_count": 0,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        result = create_bar_chart_ascii(queries)
        
        assert result == "No records added yet."
    
    def test_create_bar_chart_single_query(self):
        """Test creating chart with single query."""
        queries = [
            {
                "query": "lung cancer detection",
                "total_added": 10,
                "arxiv_count": 5,
                "pubmed_count": 3,
                "huggingface_count": 2,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        result = create_bar_chart_ascii(queries)
        
        assert "RAG STORE ADDITIONS - BAR CHART" in result
        assert "lung cancer detection" in result
        assert "10 total" in result
        assert "ArXiv: 5" in result
        assert "PubMed: 3" in result
        assert "HuggingFace: 2" in result
        assert "2025-01-01 12:00" in result
    
    def test_create_bar_chart_multiple_queries(self):
        """Test creating chart with multiple queries."""
        queries = [
            {
                "query": "query 1",
                "total_added": 20,
                "arxiv_count": 10,
                "pubmed_count": 5,
                "huggingface_count": 5,
                "timestamp": "2025-01-01T12:00:00"
            },
            {
                "query": "query 2",
                "total_added": 10,
                "arxiv_count": 5,
                "pubmed_count": 3,
                "huggingface_count": 2,
                "timestamp": "2025-01-02T12:00:00"
            }
        ]
        
        result = create_bar_chart_ascii(queries)
        
        assert " 1. query 1" in result
        assert " 2. query 2" in result
        assert "20 total" in result
        assert "10 total" in result
    
    def test_create_bar_chart_truncates_long_query(self):
        """Test that long queries are truncated."""
        queries = [
            {
                "query": "a" * 50,  # Very long query
                "total_added": 5,
                "arxiv_count": 5,
                "pubmed_count": 0,
                "huggingface_count": 0,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        result = create_bar_chart_ascii(queries)
        
        # Should be truncated with ...
        assert "..." in result
        assert "a" * 50 not in result
    
    def test_create_bar_chart_custom_width(self):
        """Test creating chart with custom width."""
        queries = [
            {
                "query": "test",
                "total_added": 10,
                "arxiv_count": 10,
                "pubmed_count": 0,
                "huggingface_count": 0,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        result = create_bar_chart_ascii(queries, max_width=20)
        
        # Should contain bars (█)
        assert "█" in result


class TestCreateBarChartHtml:
    """Test HTML bar chart creation."""
    
    def test_create_html_chart_empty(self, tmp_path):
        """Test creating HTML chart with no queries."""
        output_file = tmp_path / "chart.html"
        
        create_bar_chart_html([], output_file=str(output_file))
        
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "No queries tracked yet" in content
        assert "<!DOCTYPE html>" in content
    
    def test_create_html_chart_single_query(self, tmp_path):
        """Test creating HTML chart with single query."""
        output_file = tmp_path / "chart.html"
        queries = [
            {
                "query": "lung cancer detection",
                "total_added": 10,
                "arxiv_count": 5,
                "pubmed_count": 3,
                "huggingface_count": 2,
                "timestamp": "2025-01-01T12:00:00",
                "embeddings_provider": "openai"
            }
        ]
        
        create_bar_chart_html(queries, output_file=str(output_file))
        
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "lung cancer detection" in content
        assert "10 records" in content
        assert "ArXiv: 5" in content
        assert "PubMed: 3" in content
        assert "HuggingFace: 2" in content
        assert "2025-01-01 12:00:00" in content
    
    def test_create_html_chart_multiple_queries(self, tmp_path):
        """Test creating HTML chart with multiple queries."""
        output_file = tmp_path / "chart.html"
        queries = [
            {
                "query": "query 1",
                "total_added": 20,
                "arxiv_count": 10,
                "pubmed_count": 5,
                "huggingface_count": 5,
                "timestamp": "2025-01-01T12:00:00"
            },
            {
                "query": "query 2",
                "total_added": 10,
                "arxiv_count": 5,
                "pubmed_count": 3,
                "huggingface_count": 2,
                "timestamp": "2025-01-02T12:00:00"
            }
        ]
        
        create_bar_chart_html(queries, output_file=str(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "query 1" in content
        assert "query 2" in content
        assert "20 records" in content
        assert "10 records" in content
    
    def test_create_html_chart_includes_summary(self, tmp_path):
        """Test that HTML chart includes summary section."""
        output_file = tmp_path / "chart.html"
        queries = [
            {
                "query": "query 1",
                "total_added": 10,
                "arxiv_count": 5,
                "pubmed_count": 3,
                "huggingface_count": 2,
                "timestamp": "2025-01-01T12:00:00"
            },
            {
                "query": "query 2",
                "total_added": 15,
                "arxiv_count": 7,
                "pubmed_count": 5,
                "huggingface_count": 3,
                "timestamp": "2025-01-02T12:00:00"
            }
        ]
        
        create_bar_chart_html(queries, output_file=str(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Queries: 2" in content
        assert "Total Records: 25" in content
        assert "ArXiv: 12" in content
        assert "PubMed: 8" in content
        assert "HuggingFace: 5" in content
    
    def test_create_html_chart_utf8_encoding(self, tmp_path):
        """Test that HTML chart uses UTF-8 encoding."""
        output_file = tmp_path / "chart.html"
        queries = [
            {
                "query": "test",
                "total_added": 5,
                "arxiv_count": 5,
                "pubmed_count": 0,
                "huggingface_count": 0,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        create_bar_chart_html(queries, output_file=str(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '<meta charset="UTF-8">' in content
    
    def test_create_html_chart_css_styling(self, tmp_path):
        """Test that HTML chart includes CSS styling."""
        output_file = tmp_path / "chart.html"
        queries = [
            {
                "query": "test",
                "total_added": 5,
                "arxiv_count": 5,
                "pubmed_count": 0,
                "huggingface_count": 0,
                "timestamp": "2025-01-01T12:00:00"
            }
        ]
        
        create_bar_chart_html(queries, output_file=str(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "<style>" in content
        assert ".bar-container" in content
        assert ".source-badge" in content


class TestRAGTrackerIntegration:
    """Integration tests for RAG tracker."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from tracking to visualization."""
        tracking_file = tmp_path / "tracking.json"
        tracker = RAGTracker(tracking_file=str(tracking_file))
        
        # Track multiple queries
        tracker.track_query("lung cancer", arxiv_count=10, pubmed_count=5)
        tracker.track_query("brain tumor", pubmed_count=8, huggingface_count=3)
        tracker.track_query("skin lesion", arxiv_count=5, huggingface_count=2)
        
        # Get summary
        summary = tracker.get_summary()
        assert summary["total_queries"] == 3
        assert summary["total_records"] == 33
        
        # Get recent queries
        recent = tracker.get_recent_queries(limit=2)
        assert len(recent) == 2
        assert recent[0]["query"] == "brain tumor"
        assert recent[1]["query"] == "skin lesion"
        
        # Create ASCII chart
        ascii_chart = create_bar_chart_ascii(tracker.get_all_queries())
        assert "lung cancer" in ascii_chart
        assert "brain tumor" in ascii_chart
        assert "skin lesion" in ascii_chart
        
        # Create HTML chart
        html_file = tmp_path / "chart.html"
        create_bar_chart_html(tracker.get_all_queries(), output_file=str(html_file))
        assert html_file.exists()
        
        # Clear and verify
        tracker.clear_tracking()
        assert tracker.get_total_records() == 0
        assert len(tracker.get_all_queries()) == 0
    
    def test_persistence_across_instances(self, tmp_path):
        """Test that data persists across tracker instances."""
        tracking_file = tmp_path / "tracking.json"
        
        # Create first instance and track
        tracker1 = RAGTracker(tracking_file=str(tracking_file))
        tracker1.track_query("query 1", arxiv_count=5)
        tracker1.track_query("query 2", pubmed_count=3)
        
        # Create second instance
        tracker2 = RAGTracker(tracking_file=str(tracking_file))
        
        # Should load existing data
        assert len(tracker2.get_all_queries()) == 2
        assert tracker2.get_total_records() == 8
        
        # Add more and verify
        tracker2.track_query("query 3", huggingface_count=2)
        
        # Create third instance
        tracker3 = RAGTracker(tracking_file=str(tracking_file))
        assert len(tracker3.get_all_queries()) == 3
        assert tracker3.get_total_records() == 10
