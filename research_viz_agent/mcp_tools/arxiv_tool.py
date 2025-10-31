"""
MCP tool for searching and retrieving papers from arXiv.
"""
import arxiv
from typing import List, Dict, Optional
from datetime import datetime


class ArxivTool:
    """Tool for searching arXiv papers related to medical computer vision."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort criterion for results
            
        Returns:
            List of paper dictionaries with metadata
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        results = []
        for paper in self.client.results(search):
            results.append({
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'published': paper.published.isoformat() if paper.published else None,
                'updated': paper.updated.isoformat() if paper.updated else None,
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'entry_id': paper.entry_id,
                'doi': paper.doi,
                'primary_category': paper.primary_category,
            })
        
        return results
    
    def search_medical_cv_models(self, additional_terms: str = "") -> List[Dict]:
        """
        Search for medical computer vision papers on arXiv.
        
        Args:
            additional_terms: Additional search terms to refine the query
            
        Returns:
            List of relevant papers
        """
        base_query = 'cat:cs.CV AND (medical OR clinical OR radiology OR pathology OR diagnostic imaging)'
        if additional_terms:
            query = f"{base_query} AND ({additional_terms})"
        else:
            query = base_query
        
        return self.search_papers(query, max_results=20)


def create_arxiv_tool() -> ArxivTool:
    """Create and return an ArxivTool instance."""
    return ArxivTool()
