"""
RAG Store Query Tracker - Track and visualize additions to the RAG store.
"""
import json
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path


class RAGTracker:
    """Track queries and records added to the RAG store."""
    
    def __init__(self, tracking_file: str = "./rag_tracking.json"):
        """
        Initialize the RAG tracker.
        
        Args:
            tracking_file: Path to JSON file for storing tracking data
        """
        self.tracking_file = tracking_file
        self.data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data from file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"queries": [], "total_records": 0}
        return {"queries": [], "total_records": 0}
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.tracking_file) or '.', exist_ok=True)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def track_query(
        self,
        query: str,
        arxiv_count: int = 0,
        pubmed_count: int = 0,
        huggingface_count: int = 0,
        embeddings_provider: str = "unknown"
    ) -> None:
        """
        Track a query and the number of records added.
        
        Args:
            query: The search query
            arxiv_count: Number of ArXiv records added
            pubmed_count: Number of PubMed records added
            huggingface_count: Number of HuggingFace records added
            embeddings_provider: Provider used for embeddings
        """
        total_added = arxiv_count + pubmed_count + huggingface_count
        
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "arxiv_count": arxiv_count,
            "pubmed_count": pubmed_count,
            "huggingface_count": huggingface_count,
            "total_added": total_added,
            "embeddings_provider": embeddings_provider
        }
        
        self.data["queries"].append(query_entry)
        self.data["total_records"] += total_added
        
        self._save_tracking_data()
    
    def get_all_queries(self) -> List[Dict]:
        """Get all tracked queries."""
        return self.data.get("queries", [])
    
    def get_total_records(self) -> int:
        """Get total number of records tracked."""
        return self.data.get("total_records", 0)
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get the most recent queries."""
        queries = self.data.get("queries", [])
        return queries[-limit:] if len(queries) > limit else queries
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        queries = self.data.get("queries", [])
        
        if not queries:
            return {
                "total_queries": 0,
                "total_records": 0,
                "total_arxiv": 0,
                "total_pubmed": 0,
                "total_huggingface": 0
            }
        
        return {
            "total_queries": len(queries),
            "total_records": self.data.get("total_records", 0),
            "total_arxiv": sum(q.get("arxiv_count", 0) for q in queries),
            "total_pubmed": sum(q.get("pubmed_count", 0) for q in queries),
            "total_huggingface": sum(q.get("huggingface_count", 0) for q in queries)
        }
    
    def clear_tracking(self) -> None:
        """Clear all tracking data."""
        self.data = {"queries": [], "total_records": 0}
        self._save_tracking_data()


def create_bar_chart_ascii(
    queries: List[Dict],
    max_width: int = 60
) -> str:
    """
    Create an ASCII bar chart showing records added per query.
    
    Args:
        queries: List of query dictionaries
        max_width: Maximum width of the bars
        
    Returns:
        ASCII bar chart as string
    """
    if not queries:
        return "No queries tracked yet."
    
    # Find the maximum count for scaling
    max_count = max(q.get("total_added", 0) for q in queries)
    
    if max_count == 0:
        return "No records added yet."
    
    chart_lines = []
    chart_lines.append("\n" + "=" * 80)
    chart_lines.append("RAG STORE ADDITIONS - BAR CHART")
    chart_lines.append("=" * 80)
    chart_lines.append("")
    
    for i, query in enumerate(queries, 1):
        query_text = query.get("query", "Unknown")
        total = query.get("total_added", 0)
        arxiv = query.get("arxiv_count", 0)
        pubmed = query.get("pubmed_count", 0)
        hf = query.get("huggingface_count", 0)
        timestamp = query.get("timestamp", "")
        
        # Truncate long queries
        if len(query_text) > 40:
            query_text = query_text[:37] + "..."
        
        # Calculate bar width
        bar_width = int((total / max_count) * max_width) if max_count > 0 else 0
        bar = "█" * bar_width
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = timestamp[:16]
        else:
            time_str = "Unknown"
        
        # Add query entry
        chart_lines.append(f"{i:2}. {query_text}")
        chart_lines.append(f"    {time_str}")
        chart_lines.append(f"    {bar} {total} total")
        chart_lines.append(f"    (ArXiv: {arxiv}, PubMed: {pubmed}, HuggingFace: {hf})")
        chart_lines.append("")
    
    chart_lines.append("=" * 80)
    
    return "\n".join(chart_lines)


def create_bar_chart_html(
    queries: List[Dict],
    output_file: str = "./rag_chart.html"
) -> None:
    """
    Create an HTML bar chart showing records added per query.
    
    Args:
        queries: List of query dictionaries
        output_file: Path to output HTML file
    """
    if not queries:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Store Additions</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                p { color: #666; }
            </style>
        </head>
        <body>
            <h1>RAG Store Additions</h1>
            <p>No queries tracked yet.</p>
        </body>
        </html>
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return
    
    max_count = max(q.get("total_added", 0) for q in queries)
    
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Store Additions</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h1::before {
            content: "\\1F4CA ";
        }
        .summary {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .summary-item {
            display: inline-block;
            margin: 5px 15px 5px 0;
            font-weight: bold;
        }
        .query-item {
            margin: 20px 0;
            padding: 15px;
            border-left: 4px solid #3498db;
            background-color: #f9f9f9;
        }
        .query-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .query-time {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        .query-time::before {
            content: "\\1F550 ";
        }
        .bar-container {
            background-color: #ecf0f1;
            border-radius: 5px;
            height: 30px;
            position: relative;
            margin: 10px 0;
        }
        .bar {
            background: linear-gradient(90deg, #3498db, #2980b9);
            height: 100%;
            border-radius: 5px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }
        .details {
            font-size: 0.9em;
            color: #555;
            margin-top: 5px;
        }
        .source-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            margin-right: 8px;
            font-size: 0.85em;
        }
        .arxiv { background-color: #e74c3c; color: white; }
        .pubmed { background-color: #2ecc71; color: white; }
        .huggingface { background-color: #f39c12; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Store Additions Tracker</h1>
"""]
    
    # Add summary
    total_queries = len(queries)
    total_records = sum(q.get("total_added", 0) for q in queries)
    total_arxiv = sum(q.get("arxiv_count", 0) for q in queries)
    total_pubmed = sum(q.get("pubmed_count", 0) for q in queries)
    total_hf = sum(q.get("huggingface_count", 0) for q in queries)
    
    html_parts.append(f"""
        <div class="summary">
            <div class="summary-item">Queries: {total_queries}</div>
            <div class="summary-item">Total Records: {total_records}</div>
            <div class="summary-item">ArXiv: {total_arxiv}</div>
            <div class="summary-item">PubMed: {total_pubmed}</div>
            <div class="summary-item">HuggingFace: {total_hf}</div>
        </div>
""")
    
    # Add query entries
    for i, query in enumerate(queries, 1):
        query_text = query.get("query", "Unknown")
        total = query.get("total_added", 0)
        arxiv = query.get("arxiv_count", 0)
        pubmed = query.get("pubmed_count", 0)
        hf = query.get("huggingface_count", 0)
        timestamp = query.get("timestamp", "")
        
        # Calculate bar width percentage
        bar_width_pct = (total / max_count * 100) if max_count > 0 else 0
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = timestamp
        else:
            time_str = "Unknown"
        
        html_parts.append(f"""
        <div class="query-item">
            <div class="query-title">{i}. {query_text}</div>
            <div class="query-time">{time_str}</div>
            <div class="bar-container">
                <div class="bar" style="width: {bar_width_pct}%;">
                    {total} records
                </div>
            </div>
            <div class="details">
                <span class="source-badge arxiv">ArXiv: {arxiv}</span>
                <span class="source-badge pubmed">PubMed: {pubmed}</span>
                <span class="source-badge huggingface">HuggingFace: {hf}</span>
            </div>
        </div>
""")
    
    html_parts.append("""
    </div>
</body>
</html>
""")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("".join(html_parts))
