"""
Tests for CSV export functionality.
"""
import os
import csv
import tempfile
import pytest
from research_viz_agent.utils.csv_export import (
    export_rag_results_to_csv,
    export_research_results_to_csv
)


class TestCSVExport:
    """Test CSV export utilities."""
    
    def test_export_rag_results_basic(self):
        """Test basic RAG results export."""
        rag_results = {
            'query': 'test query',
            'results': [
                {
                    'content': 'Test content',
                    'metadata': {
                        'source': 'arxiv',
                        'type': 'paper',
                        'title': 'Test Paper',
                        'url': 'http://example.com',
                        'authors': 'John Doe, Jane Smith',
                        'categories': 'cs.CV',
                        'published': '2023-01-01',
                        'entry_id': 'test123',
                        'indexed_at': '2023-01-02T12:00:00',
                        'query': 'original query'
                    },
                    'source': 'arxiv',
                    'type': 'paper',
                    'title': 'Test Paper',
                    'url': 'http://example.com'
                }
            ],
            'total_count': 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            export_rag_results_to_csv(rag_results, temp_file)
            
            # Verify file was created
            assert os.path.exists(temp_file)
            
            # Read and verify contents
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 1
                assert rows[0]['source'] == 'arxiv'
                assert rows[0]['type'] == 'paper'
                assert rows[0]['title'] == 'Test Paper'
                assert rows[0]['url'] == 'http://example.com'
                assert rows[0]['authors'] == 'John Doe, Jane Smith'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_rag_results_with_error(self):
        """Test that export fails gracefully with error results."""
        rag_results = {
            'error': 'Test error',
            'results': [],
            'total_count': 0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Cannot export results with error"):
                export_rag_results_to_csv(rag_results, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_rag_results_empty(self):
        """Test that export fails with empty results."""
        rag_results = {
            'query': 'test',
            'results': [],
            'total_count': 0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="No results to export"):
                export_rag_results_to_csv(rag_results, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_rag_results_pubmed(self):
        """Test exporting PubMed RAG results."""
        rag_results = {
            'query': 'test query',
            'results': [
                {
                    'content': 'Test content with Abstract: This is the abstract\n\nMore content',
                    'metadata': {
                        'source': 'pubmed',
                        'type': 'paper',
                        'title': 'PubMed Paper',
                        'url': 'https://pubmed.ncbi.nlm.nih.gov/12345/',
                        'authors': 'Smith J',
                        'journal': 'Test Journal',
                        'publication_date': '2023-05',
                        'pmid': '12345',
                        'mesh_terms': 'term1, term2',
                        'indexed_at': '2023-01-02T12:00:00'
                    },
                    'source': 'pubmed',
                    'type': 'paper',
                    'title': 'PubMed Paper',
                    'url': 'https://pubmed.ncbi.nlm.nih.gov/12345/'
                }
            ],
            'total_count': 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            export_rag_results_to_csv(rag_results, temp_file)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 1
                assert rows[0]['source'] == 'pubmed'
                assert rows[0]['pmid'] == '12345'
                assert rows[0]['journal'] == 'Test Journal'
                assert 'This is the abstract' in rows[0]['abstract']
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_research_results_arxiv(self):
        """Test exporting regular research results with ArXiv papers."""
        research_results = {
            'arxiv_results': [
                {
                    'title': 'ArXiv Paper',
                    'authors': ['John Doe', 'Jane Smith'],
                    'summary': 'This is the abstract',
                    'pdf_url': 'http://arxiv.org/pdf/1234.56789',
                    'published': '2023-01-01',
                    'categories': ['cs.CV', 'cs.AI'],
                    'entry_id': 'arxiv:1234.56789',
                    'primary_category': 'cs.CV'
                }
            ],
            'pubmed_results': [],
            'huggingface_results': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            export_research_results_to_csv(research_results, temp_file)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 1
                assert rows[0]['source'] == 'arxiv'
                assert rows[0]['title'] == 'ArXiv Paper'
                assert rows[0]['authors'] == 'John Doe, Jane Smith'
                assert rows[0]['abstract'] == 'This is the abstract'
                assert rows[0]['categories'] == 'cs.CV, cs.AI'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_research_results_huggingface(self):
        """Test exporting HuggingFace model results."""
        research_results = {
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': [
                {
                    'model_id': 'test/model',
                    'author': 'testuser',
                    'tags': ['pytorch', 'vision'],
                    'downloads': 1000,
                    'likes': 50,
                    'model_card_url': 'https://huggingface.co/test/model',
                    'library_name': 'pytorch',
                    'pipeline_tag': 'image-classification',
                    'created_at': '2023-01-01',
                    'last_modified': '2023-01-02'
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            export_research_results_to_csv(research_results, temp_file)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 1
                assert rows[0]['source'] == 'huggingface'
                assert rows[0]['type'] == 'model'
                assert rows[0]['model_id'] == 'test/model'
                assert rows[0]['downloads'] == '1000'
                assert rows[0]['likes'] == '50'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_research_results_mixed(self):
        """Test exporting mixed results from multiple sources."""
        research_results = {
            'arxiv_results': [
                {
                    'title': 'ArXiv Paper',
                    'authors': ['Author A'],
                    'summary': 'Abstract A',
                    'pdf_url': 'http://arxiv.org/pdf/1',
                    'published': '2023-01-01',
                    'categories': ['cs.CV'],
                    'entry_id': 'arxiv:1',
                    'primary_category': 'cs.CV'
                }
            ],
            'pubmed_results': [
                {
                    'title': 'PubMed Paper',
                    'authors': ['Author B'],
                    'abstract': 'Abstract B',
                    'pmid': '12345',
                    'journal': 'Journal A',
                    'publication_date': '2023-02-01',
                    'mesh_terms': ['term1'],
                    'keywords': ['key1']
                }
            ],
            'huggingface_results': [
                {
                    'model_id': 'test/model',
                    'author': 'Author C',
                    'tags': ['tag1'],
                    'downloads': 100,
                    'likes': 10,
                    'model_card_url': 'http://hf.co/test',
                    'library_name': 'pytorch',
                    'pipeline_tag': 'classification',
                    'created_at': '2023-03-01',
                    'last_modified': '2023-03-02'
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            export_research_results_to_csv(research_results, temp_file)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                assert len(rows) == 3
                assert rows[0]['source'] == 'arxiv'
                assert rows[1]['source'] == 'pubmed'
                assert rows[2]['source'] == 'huggingface'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_export_research_results_empty(self):
        """Test that export fails with no results."""
        research_results = {
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="No results to export"):
                export_research_results_to_csv(research_results, temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
