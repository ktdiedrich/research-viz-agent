"""
CSV export utilities for RAG search results and research data.
"""
import csv
from typing import Dict
def export_rag_results_to_csv(rag_results: Dict, output_file: str) -> None:
    """
    Export RAG search results to CSV file.
    
    Args:
        rag_results: Results from agent.search_rag()
        output_file: Path to output CSV file
    """
    if 'error' in rag_results:
        raise ValueError(f"Cannot export results with error: {rag_results['error']}")
    
    if not rag_results['results']:
        raise ValueError("No results to export")
    
    # Define CSV columns
    fieldnames = [
        'source',
        'type',
        'title',
        'url',
        'authors',
        'abstract',
        'journal',
        'publication_date',
        'categories',
        'tags',
        'downloads',
        'likes',
        'pmid',
        'entry_id',
        'model_id',
        'indexed_at',
        'query'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for result in rag_results['results']:
            metadata = result['metadata']
            
            # Extract abstract from content if not in metadata
            abstract = metadata.get('abstract', '')
            if not abstract and 'content' in result:
                content = result['content']
                # Try to extract abstract from content
                if 'Abstract:' in content:
                    abstract_start = content.index('Abstract:') + len('Abstract:')
                    abstract_end = content.find('\n\n', abstract_start)
                    if abstract_end > abstract_start:
                        abstract = content[abstract_start:abstract_end].strip()
            
            # Prepare row data
            row = {
                'source': metadata.get('source', ''),
                'type': metadata.get('type', ''),
                'title': metadata.get('title', ''),
                'url': metadata.get('url', ''),
                'authors': metadata.get('authors', ''),
                'abstract': abstract,
                'journal': metadata.get('journal', ''),
                'publication_date': metadata.get('publication_date', metadata.get('published', '')),
                'categories': metadata.get('categories', ''),
                'tags': metadata.get('tags', ''),
                'downloads': metadata.get('downloads', ''),
                'likes': metadata.get('likes', ''),
                'pmid': metadata.get('pmid', ''),
                'entry_id': metadata.get('entry_id', ''),
                'model_id': metadata.get('model_id', ''),
                'indexed_at': metadata.get('indexed_at', ''),
                'query': metadata.get('query', rag_results.get('query', ''))
            }
            
            writer.writerow(row)


def export_research_results_to_csv(research_results: Dict, output_file: str) -> None:
    """
    Export regular research results to CSV file.
    
    Args:
        research_results: Results from agent.research()
        output_file: Path to output CSV file
    """
    # Define CSV columns
    fieldnames = [
        'source',
        'type',
        'title',
        'url',
        'authors',
        'abstract',
        'journal',
        'publication_date',
        'categories',
        'tags',
        'downloads',
        'likes',
        'pmid',
        'entry_id',
        'model_id',
        'primary_category',
        'mesh_terms',
        'keywords',
        'library',
        'pipeline_tag',
        'created_at',
        'last_modified'
    ]
    
    rows = []
    
    # Process ArXiv results
    for paper in research_results.get('arxiv_results', []):
        row = {
            'source': 'arxiv',
            'type': 'paper',
            'title': paper.get('title', ''),
            'url': paper.get('pdf_url', paper.get('entry_url', '')),
            'authors': ', '.join(paper.get('authors', [])),
            'abstract': paper.get('summary', ''),
            'publication_date': paper.get('published', ''),
            'categories': ', '.join(paper.get('categories', [])),
            'entry_id': paper.get('entry_id', ''),
            'primary_category': paper.get('primary_category', '')
        }
        rows.append(row)
    
    # Process PubMed results
    for paper in research_results.get('pubmed_results', []):
        row = {
            'source': 'pubmed',
            'type': 'paper',
            'title': paper.get('title', ''),
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid', '')}/",
            'authors': ', '.join(paper.get('authors', [])),
            'abstract': paper.get('abstract', ''),
            'journal': paper.get('journal', ''),
            'publication_date': paper.get('publication_date', ''),
            'pmid': paper.get('pmid', ''),
            'mesh_terms': ', '.join(paper.get('mesh_terms', [])),
            'keywords': ', '.join(paper.get('keywords', []))
        }
        rows.append(row)
    
    # Process HuggingFace results
    for model in research_results.get('huggingface_results', []):
        row = {
            'source': 'huggingface',
            'type': 'model',
            'title': model.get('model_id', ''),
            'url': model.get('model_card_url', ''),
            'authors': model.get('author', ''),
            'tags': ', '.join(model.get('tags', [])),
            'downloads': model.get('downloads', ''),
            'likes': model.get('likes', ''),
            'model_id': model.get('model_id', ''),
            'library': model.get('library_name', ''),
            'pipeline_tag': model.get('pipeline_tag', ''),
            'created_at': model.get('created_at', ''),
            'last_modified': model.get('last_modified', '')
        }
        rows.append(row)
    
    if not rows:
        raise ValueError("No results to export")
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
