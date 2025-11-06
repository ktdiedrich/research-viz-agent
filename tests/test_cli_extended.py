"""
Extended unit tests for CLI module to increase coverage.

Tests focus on:
- Agent server (serve command)
- RAG tracking functionality
- CSV export integration
- Edge cases and error handling
"""
import pytest
from unittest.mock import MagicMock, patch, mock_open, Mock
from io import StringIO
import json

from research_viz_agent.cli import main


class TestCLIServeCommand:
    """Test CLI serve command for agent server."""
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve'])
    def test_serve_command_default_options(self, mock_validate, mock_create_server):
        """Test serve command with default options."""
        mock_validate.return_value = (True, "GitHub token configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            # Verify server startup messages
            assert '🤖 Starting Medical CV Research Agent Server' in output
            assert 'Host: 0.0.0.0' in output
            assert 'Port: 8000' in output
            assert 'LLM Provider: github' in output
            assert 'Base URL: http://0.0.0.0:8000' in output
            assert 'Status: http://0.0.0.0:8000/status' in output
            assert 'Docs: http://0.0.0.0:8000/docs' in output
            assert 'Health: http://0.0.0.0:8000/health' in output
            
            # Verify server was created with correct args
            mock_create_server.assert_called_once()
            call_args = mock_create_server.call_args
            assert call_args[1]['llm_provider'] == 'github'
            assert call_args[1]['host'] == '0.0.0.0'
            assert call_args[1]['port'] == 8000
            assert call_args[1]['enable_rag'] is True
            
            # Verify server.run() was called
            mock_server.run.assert_called_once()
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve', '--port', '9000', '--host', 'localhost'])
    def test_serve_command_custom_options(self, mock_validate, mock_create_server):
        """Test serve command with custom host and port."""
        mock_validate.return_value = (True, "Configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'Host: localhost' in output
            assert 'Port: 9000' in output
            assert 'http://localhost:9000' in output
            
            call_args = mock_create_server.call_args
            assert call_args[1]['host'] == 'localhost'
            assert call_args[1]['port'] == 9000
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve', '--llm-provider', 'openai', '--model', 'gpt-4o-mini'])
    def test_serve_command_with_llm_options(self, mock_validate, mock_create_server):
        """Test serve command with LLM provider options."""
        mock_validate.return_value = (True, "Configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'LLM Provider: openai' in output
            
            call_args = mock_create_server.call_args
            assert call_args[1]['llm_provider'] == 'openai'
            assert call_args[1]['model_name'] == 'gpt-4o-mini'
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve', '--no-rag', '--max-results', '50', '--temperature', '0.5'])
    def test_serve_command_with_agent_options(self, mock_validate, mock_create_server):
        """Test serve command with agent configuration options."""
        mock_validate.return_value = (True, "Configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'Max Results: 50' in output
            assert 'RAG Enabled: False' in output
            assert 'Temperature: 0.5' in output
            
            call_args = mock_create_server.call_args
            assert call_args[1]['enable_rag'] is False
            assert call_args[1]['max_results'] == 50
            assert call_args[1]['temperature'] == 0.5
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve', '--no-summary'])
    def test_serve_command_no_summary_fallback(self, mock_validate, mock_create_server):
        """Test serve command falls back to 'none' provider with --no-summary."""
        mock_validate.return_value = (True, "Configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'LLM Provider: none' in output
            
            call_args = mock_create_server.call_args
            assert call_args[1]['llm_provider'] == 'none'
    
    @patch('research_viz_agent.agent_protocol.server.create_agent_server')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'serve', '--llm-provider', 'github'])
    def test_serve_command_invalid_provider_fallback(self, mock_validate, mock_create_server):
        """Test serve command falls back when provider config is invalid."""
        mock_validate.return_value = (False, "GitHub token not configured")
        mock_server = MagicMock()
        mock_create_server.return_value = mock_server
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert '⚠ GitHub token not configured' in output
            assert 'Falling back to no-summary mode' in output
            assert 'LLM Provider: none' in output
            
            call_args = mock_create_server.call_args
            assert call_args[1]['llm_provider'] == 'none'


class TestCLIRAGTracking:
    """Test CLI RAG tracking functionality."""
    
    @patch('research_viz_agent.cli.RAGTracker')
    @patch('sys.argv', ['cli.py', '--tracking-summary'])
    def test_tracking_summary(self, mock_tracker_class):
        """Test --tracking-summary displays summary."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_summary.return_value = {
            'total_queries': 10,
            'total_records': 150,
            'total_arxiv': 60,
            'total_pubmed': 70,
            'total_huggingface': 20
        }
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'RAG STORE TRACKING SUMMARY' in output
            assert 'Total Queries: 10' in output
            assert 'Total Records: 150' in output
            assert '- ArXiv: 60' in output
            assert '- PubMed: 70' in output
            assert '- HuggingFace: 20' in output
    
    @patch('research_viz_agent.cli.RAGTracker')
    @patch('research_viz_agent.cli.create_bar_chart_ascii')
    @patch('sys.argv', ['cli.py', '--show-tracking'])
    def test_show_tracking(self, mock_chart, mock_tracker_class):
        """Test --show-tracking displays chart."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_all_queries.return_value = [
            {'query': 'test1', 'records_added': 10},
            {'query': 'test2', 'records_added': 20}
        ]
        mock_chart.return_value = "ASCII CHART HERE"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'ASCII CHART HERE' in output
            mock_tracker.get_all_queries.assert_called_once()
            mock_chart.assert_called_once()
    
    @patch('research_viz_agent.cli.RAGTracker')
    @patch('sys.argv', ['cli.py', '--tracking-summary', '--llm-provider', 'github'])
    def test_tracking_with_github_provider(self, mock_tracker_class):
        """Test tracking uses correct directory for GitHub provider."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_summary.return_value = {
            'total_queries': 0,
            'total_records': 0,
            'total_arxiv': 0,
            'total_pubmed': 0,
            'total_huggingface': 0
        }
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            # Verify tracker was initialized with GitHub directory
            call_args = mock_tracker_class.call_args
            tracking_file = call_args[1]['tracking_file']
            assert '_github' in tracking_file
    
    @patch('research_viz_agent.cli.RAGTracker')
    @patch('sys.argv', ['cli.py', '--tracking-summary', '--llm-provider', 'openai'])
    def test_tracking_with_openai_provider(self, mock_tracker_class):
        """Test tracking uses correct directory for OpenAI provider."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_summary.return_value = {
            'total_queries': 0,
            'total_records': 0,
            'total_arxiv': 0,
            'total_pubmed': 0,
            'total_huggingface': 0
        }
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            # Verify tracker was initialized with standard directory
            call_args = mock_tracker_class.call_args
            tracking_file = call_args[1]['tracking_file']
            assert 'chroma_db/rag_tracking.json' in tracking_file
    
    @patch('research_viz_agent.cli.RAGTracker')
    @patch('sys.argv', ['cli.py', '--tracking-summary', '--rag-dir', '/custom/path'])
    def test_tracking_with_custom_rag_dir(self, mock_tracker_class):
        """Test tracking uses custom RAG directory."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_summary.return_value = {
            'total_queries': 0,
            'total_records': 0,
            'total_arxiv': 0,
            'total_pubmed': 0,
            'total_huggingface': 0
        }
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_args = mock_tracker_class.call_args
            tracking_file = call_args[1]['tracking_file']
            assert '/custom/path_github' in tracking_file


class TestCLICSVExport:
    """Test CSV export functionality in CLI."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.export_research_results_to_csv')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'lung cancer', '--csv', 'output.csv'])
    def test_research_with_csv_export(self, mock_validate, mock_export, mock_agent_class):
        """Test research query with CSV export."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.research.return_value = {
            'summary': 'Test summary',
            'total_papers': 10,
            'arxiv_results': []
        }
        mock_agent.format_results.return_value = "Formatted results"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            # Verify export was called
            mock_export.assert_called_once()
            call_args = mock_export.call_args
            assert call_args[0][1] == 'output.csv'
            
            # Verify success message
            assert 'Results exported to output.csv' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.export_research_results_to_csv')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'brain tumor', '--csv', 'test.csv'])
    def test_csv_export_error_handling(self, mock_validate, mock_export, mock_agent_class):
        """Test CSV export error handling."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.research.return_value = {
            'summary': 'Test',
            'total_papers': 5,
            'arxiv_results': []
        }
        mock_agent.format_results.return_value = "Results"
        
        # Simulate export failure
        mock_export.side_effect = Exception("Export failed")
        
        with patch('sys.stdout', new=StringIO()):
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                main()
                error_output = mock_stderr.getvalue()
                
                # Verify error message
                assert 'Warning: Failed to export CSV: Export failed' in error_output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.export_rag_results_to_csv')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-search', 'test query', '--csv', 'rag_output.csv'])
    def test_rag_search_with_csv_export(self, mock_validate, mock_export, mock_agent_class):
        """Test RAG search with CSV export."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.search_rag.return_value = {
            'total_count': 5,
            'results': []
        }
        mock_agent.format_rag_results.return_value = "RAG results"
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            # Verify RAG export was called
            mock_export.assert_called_once()
            call_args = mock_export.call_args
            assert call_args[0][1] == 'rag_output.csv'
            
            assert 'Results exported to rag_output.csv' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.export_rag_results_to_csv')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-search', 'query', '--csv', 'output.csv'])
    def test_rag_csv_export_error_handling(self, mock_validate, mock_export, mock_agent_class):
        """Test RAG CSV export error handling."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.search_rag.return_value = {'total_count': 0, 'results': []}
        mock_agent.format_rag_results.return_value = "Results"
        
        # Simulate export failure
        mock_export.side_effect = IOError("Permission denied")
        
        with patch('sys.stdout', new=StringIO()):
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                main()
                error_output = mock_stderr.getvalue()
                
                assert 'Warning: Failed to export CSV: Permission denied' in error_output


class TestCLIEdgeCases:
    """Test edge cases and corner scenarios."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--rag-dir', '/custom/db'])
    def test_custom_rag_directory(self, mock_validate, mock_agent_class):
        """Test using custom RAG directory."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.research.return_value = {
            'summary': 'Test',
            'total_papers': 0,
            'arxiv_results': []
        }
        mock_agent.format_results.return_value = "Results"
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            # Verify agent was created with custom directory
            call_args = mock_agent_class.call_args
            assert call_args[1]['rag_persist_dir'] == '/custom/db'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--email', 'custom@email.com'])
    def test_custom_pubmed_email(self, mock_validate, mock_agent_class):
        """Test custom PubMed email."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.research.return_value = {
            'summary': 'Test',
            'total_papers': 0,
            'arxiv_results': []
        }
        mock_agent.format_results.return_value = "Results"
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_args = mock_agent_class.call_args
            assert call_args[1]['pubmed_email'] == 'custom@email.com'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--display-results', '5'])
    def test_display_results_limit(self, mock_validate, mock_agent_class):
        """Test display results limit parameter."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.research.return_value = {
            'summary': 'Test',
            'total_papers': 10,
            'arxiv_results': []
        }
        mock_agent.format_results.return_value = "Results"
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            # Verify format_results was called with limit
            call_args = mock_agent.format_results.call_args
            assert call_args[1]['display_limit'] == 5


class TestCLIRAGStatsExtended:
    """Extended tests for RAG stats functionality."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-stats'])
    def test_rag_stats_with_error(self, mock_validate, mock_agent_class):
        """Test RAG stats when there's an error."""
        mock_validate.return_value = (True, "Configured")
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.get_rag_stats.return_value = {
            'error': 'Database not found'
        }
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'Error: Database not found' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
