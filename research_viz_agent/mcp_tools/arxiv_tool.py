"""
MCP tool for searching and retrieving papers from arXiv.
"""
import arxiv
from typing import List, Dict
import mcp.types as types
from mcp.server import Server


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
    
    def search_medical_cv_models(self, additional_terms: str = "", max_results: int = 20) -> List[Dict]:
        """
        Search for medical computer vision papers on arXiv.
        
        Args:
            additional_terms: Additional search terms to refine the query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant papers
        """
        base_query = 'cat:cs.CV AND (medical OR clinical OR radiology OR pathology OR diagnostic imaging)'
        if additional_terms:
            query = f"{base_query} AND ({additional_terms})"
        else:
            query = base_query
        
        return self.search_papers(query, max_results=max_results)


# MCP Server instance
server = Server("arxiv-tool")

# Tool instance
arxiv_tool = ArxivTool()

@server.call_tool()
async def search_arxiv_papers(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for papers on arXiv with specified query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10)
    """
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 10)
    
    if not query:
        return [types.TextContent(
            type="text",
            text="Error: Query parameter is required"
        )]
    
    try:
        results = arxiv_tool.search_papers(query, max_results)
        
        # Format results as text
        formatted_results = []
        for i, paper in enumerate(results, 1):
            paper_text = f"""
{i}. {paper['title']}
   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}
   Published: {paper['published']}
   Categories: {', '.join(paper['categories'])}
   Summary: {paper['summary'][:300]}...
   URL: {paper['pdf_url']}
   
"""
            formatted_results.append(paper_text)
        
        return [types.TextContent(
            type="text", 
            text=f"Found {len(results)} papers:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching arXiv: {str(e)}"
        )]

@server.call_tool()
async def search_medical_cv_papers(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for medical computer vision papers on arXiv.
    
    Args:
        additional_terms: Additional search terms to refine the query (optional)
    """
    additional_terms = arguments.get("additional_terms", "")
    
    try:
        results = arxiv_tool.search_medical_cv_models(additional_terms)
        
        # Format results as text
        formatted_results = []
        for i, paper in enumerate(results, 1):
            paper_text = f"""
{i}. {paper['title']}
   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}
   Published: {paper['published']}
   Categories: {', '.join(paper['categories'])}
   Summary: {paper['summary'][:300]}...
   URL: {paper['pdf_url']}
   
"""
            formatted_results.append(paper_text)
        
        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} medical CV papers:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching medical CV papers: {str(e)}"
        )]

def create_arxiv_tool() -> ArxivTool:
    """Create and return an ArxivTool instance."""
    return ArxivTool()

def get_arxiv_server() -> Server:
    """Get the MCP server instance for arXiv tools."""
    return server
