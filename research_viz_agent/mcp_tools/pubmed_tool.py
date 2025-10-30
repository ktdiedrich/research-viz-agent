"""
MCP tool for searching and retrieving papers from PubMed.
"""
from Bio import Entrez, Medline
from typing import List, Dict, Optional
import time


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
    
    def search_medical_cv_models(self, additional_terms: str = "") -> List[Dict]:
        """
        Search for medical computer vision and AI model papers on PubMed.
        
        Args:
            additional_terms: Additional search terms to refine the query
            
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
        
        return self.search_papers(query, max_results=20)


def create_pubmed_tool(email: str = "research@example.com") -> PubMedTool:
    """Create and return a PubMedTool instance."""
    return PubMedTool(email=email)
