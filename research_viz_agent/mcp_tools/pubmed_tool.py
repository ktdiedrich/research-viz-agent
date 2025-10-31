"""
MCP tool for searching and retrieving papers from PubMed.
"""
from Bio import Entrez, Medline
from typing import List, Dict, Optional
import time
import mcp.types as types
from mcp.server import Server


class PubMedTool:
    """Tool for searching PubMed papers related to medical computer vision."""
    
    def __init__(self, email: str = "research@example.com"):
        """
        Initialize PubMed tool.
        
        Args:
            email: Email for Entrez API (required by NCBI)
        """
        Entrez.email = email
    
    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance"
    ) -> List[Dict]:
        """
        Search for papers on PubMed.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort: Sort order (relevance, pub_date, etc.)
            
        Returns:
            List of paper dictionaries with metadata
        """
        # Search PubMed
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort=sort
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                return []
            
            # Fetch details for the papers
            # Add small delay to respect NCBI rate limits
            time.sleep(0.34)  # ~3 requests per second
            
            handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="medline",
                retmode="text"
            )
            records = Medline.parse(handle)
            
            results = []
            for record in records:
                results.append({
                    'pmid': record.get('PMID', ''),
                    'title': record.get('TI', ''),
                    'authors': record.get('AU', []),
                    'abstract': record.get('AB', ''),
                    'journal': record.get('TA', ''),
                    'publication_date': record.get('DP', ''),
                    'keywords': record.get('OT', []),
                    'mesh_terms': record.get('MH', []),
                    'doi': record.get('LID', ''),
                })
            
            handle.close()
            return results
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def search_medical_cv_models(self, additional_terms: str = "", max_results: int = 20) -> List[Dict]:
        """
        Search for medical computer vision and AI model papers on PubMed.
        
        Args:
            additional_terms: Additional search terms to refine the query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant papers
        """
        base_query = (
            '(("computer vision"[Title/Abstract] OR "deep learning"[Title/Abstract] OR '
            '"convolutional neural network"[Title/Abstract] OR "AI model"[Title/Abstract]) AND '
            '("medical imaging"[Title/Abstract] OR "radiology"[Title/Abstract] OR '
            '"pathology"[Title/Abstract] OR "diagnostic imaging"[Title/Abstract]))'
        )
        
        if additional_terms:
            query = f"{base_query} AND ({additional_terms})"
        else:
            query = base_query
        
        return self.search_papers(query, max_results=max_results)


# MCP Server instance
server = Server("pubmed-tool")

# Tool instance
pubmed_tool = PubMedTool()

@server.call_tool()
async def search_pubmed_papers(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for papers on PubMed with specified query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10)
        sort: Sort order - 'relevance', 'pub_date', etc. (default: 'relevance')
    """
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 10)
    sort = arguments.get("sort", "relevance")
    
    if not query:
        return [types.TextContent(
            type="text",
            text="Error: Query parameter is required"
        )]
    
    try:
        results = pubmed_tool.search_papers(query, max_results, sort)
        
        if not results:
            return [types.TextContent(
                type="text",
                text="No papers found matching the search criteria."
            )]
        
        # Format results as text
        formatted_results = []
        for i, paper in enumerate(results, 1):
            # Format authors (limit to first 3)
            authors_str = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors_str += f" et al. ({len(paper['authors'])} total)"
            
            # Truncate abstract for readability
            abstract = paper['abstract'][:400] + "..." if len(paper['abstract']) > 400 else paper['abstract']
            
            paper_text = f"""
{i}. {paper['title']}
   PMID: {paper['pmid']}
   Authors: {authors_str}
   Journal: {paper['journal']}
   Published: {paper['publication_date']}
   DOI: {paper['doi']}
   Abstract: {abstract}
   MeSH Terms: {', '.join(paper['mesh_terms'][:5])}{'...' if len(paper['mesh_terms']) > 5 else ''}
   URL: https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/
   
"""
            formatted_results.append(paper_text)
        
        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} PubMed papers:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching PubMed: {str(e)}"
        )]

@server.call_tool()
async def search_medical_cv_papers_pubmed(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for medical computer vision and AI model papers on PubMed.
    
    Args:
        additional_terms: Additional search terms to refine the query (optional)
    """
    additional_terms = arguments.get("additional_terms", "")
    
    try:
        results = pubmed_tool.search_medical_cv_models(additional_terms)
        
        if not results:
            return [types.TextContent(
                type="text",
                text="No medical computer vision papers found in PubMed."
            )]
        
        # Format results as text
        formatted_results = []
        for i, paper in enumerate(results, 1):
            # Format authors (limit to first 3)
            authors_str = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors_str += f" et al. ({len(paper['authors'])} total)"
            
            # Truncate abstract for readability
            abstract = paper['abstract'][:400] + "..." if len(paper['abstract']) > 400 else paper['abstract']
            
            paper_text = f"""
{i}. {paper['title']}
   PMID: {paper['pmid']}
   Authors: {authors_str}
   Journal: {paper['journal']}
   Published: {paper['publication_date']}
   DOI: {paper['doi']}
   Abstract: {abstract}
   MeSH Terms: {', '.join(paper['mesh_terms'][:5])}{'...' if len(paper['mesh_terms']) > 5 else ''}
   URL: https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/
   
"""
            formatted_results.append(paper_text)
        
        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} medical CV papers in PubMed:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching medical CV papers: {str(e)}"
        )]

@server.call_tool()
async def get_paper_by_pmid(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Get detailed information about a specific paper by PMID.
    
    Args:
        pmid: PubMed ID of the paper
    """
    pmid = arguments.get("pmid", "")
    
    if not pmid:
        return [types.TextContent(
            type="text",
            text="Error: PMID parameter is required"
        )]
    
    try:
        # Search for the specific PMID
        results = pubmed_tool.search_papers(f"{pmid}[PMID]", max_results=1)
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"Paper with PMID {pmid} not found."
            )]
        
        paper = results[0]
        
        # Format detailed paper information
        authors_str = ', '.join(paper['authors']) if paper['authors'] else 'N/A'
        keywords_str = ', '.join(paper['keywords']) if paper['keywords'] else 'N/A'
        mesh_str = ', '.join(paper['mesh_terms']) if paper['mesh_terms'] else 'N/A'
        
        details = f"""
Paper Details for PMID: {paper['pmid']}

Title: {paper['title']}

Authors: {authors_str}

Journal: {paper['journal']}
Published: {paper['publication_date']}
DOI: {paper['doi']}

Abstract:
{paper['abstract']}

Keywords: {keywords_str}

MeSH Terms: {mesh_str}

PubMed URL: https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/
"""
        
        return [types.TextContent(
            type="text",
            text=details
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error retrieving paper details: {str(e)}"
        )]

def create_pubmed_tool(email: str = "research@example.com") -> PubMedTool:
    """Create and return a PubMedTool instance."""
    return PubMedTool(email=email)

def get_pubmed_server() -> Server:
    """Get the MCP server instance for PubMed tools."""
    return server
