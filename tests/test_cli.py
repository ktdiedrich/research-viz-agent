"""
Unit tests for the CLI module.
"""
import pytest
from unittest.mock import MagicMock, patch, mock_open
from io import StringIO

from research_viz_agent.cli import main


class TestCLIListModels:
    """Test CLI --list-models functionality."""
    
    @patch('research_viz_agent.cli.LLMFactory.get_available_models')
    @patch('sys.argv', ['cli.py', '--list-models', 'openai'])
    def test_list_openai_models(self, mock_get_models):
        """Test listing OpenAI models."""
        mock_get_models.return_value = {
            'gpt-4o': {
                'description': 'GPT-4 Optimized',
                'context_window': 128000,
                'cost_per_1k_tokens': {'input': 0.0025, 'output': 0.01}
            },
            'gpt-4o-mini': {
                'description': 'Mini model',
                'context_window': 128000,
                'cost_per_1k_tokens': {'input': 0.00015, 'output': 0.0006}
            }
        }
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'OPENAI Available Models:' in output
            assert 'gpt-4o' in output
            assert 'gpt-4o-mini' in output
            assert 'GPT-4 Optimized' in output
            assert '128,000 tokens' in output
    
    @patch('research_viz_agent.cli.LLMFactory.get_available_models')
    @patch('sys.argv', ['cli.py', '--list-models', 'github'])
    def test_list_github_models(self, mock_get_models):
        """Test listing GitHub models."""
        mock_get_models.return_value = {
            'gpt-4o': {
                'description': 'GitHub hosted GPT-4',
                'provider': 'github',
                'cost': 'Free with GitHub Pro'
            }
        }
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'GITHUB Available Models:' in output
            assert 'gpt-4o' in output
            assert 'Free with GitHub Pro' in output


class TestCLIProviderInfo:
    """Test CLI --provider-info functionality."""
    
    @patch('research_viz_agent.cli.LLMFactory.get_provider_info')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--provider-info', 'openai'])
    def test_provider_info_openai_configured(self, mock_validate, mock_get_info):
        """Test provider info for configured OpenAI."""
        mock_get_info.return_value = {
            'name': 'OpenAI',
            'description': 'OpenAI API',
            'cost': 'Paid API',
            'setup_url': 'https://platform.openai.com',
            'env_var': 'OPENAI_API_KEY',
            'models': ['gpt-4o', 'gpt-4o-mini']
        }
        mock_validate.return_value = (True, "API key configured")
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'OpenAI Provider Information:' in output
            assert 'OpenAI API' in output
            assert 'OPENAI_API_KEY' in output
            assert 'Configuration Status: ✓' in output
    
    @patch('research_viz_agent.cli.LLMFactory.get_provider_info')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--provider-info', 'github'])
    def test_provider_info_github_not_configured(self, mock_validate, mock_get_info):
        """Test provider info for unconfigured GitHub."""
        mock_get_info.return_value = {
            'name': 'GitHub Models',
            'description': 'Free models',
            'cost': 'Free with GitHub Pro',
            'setup_url': 'https://github.com',
            'env_var': 'GITHUB_TOKEN',
            'requirements': 'GitHub Pro account',
            'models': ['gpt-4o', 'Llama-3.2']
        }
        mock_validate.return_value = (False, "Token not configured")
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'GitHub Models Provider Information:' in output
            assert 'Free with GitHub Pro' in output
            assert 'Configuration Status: ✗' in output
            assert 'Token not configured' in output


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    @patch('sys.argv', ['cli.py'])
    def test_missing_query_error(self):
        """Test error when query is missing."""
        with pytest.raises(SystemExit):
            with patch('sys.stderr', new=StringIO()):
                main()
    
    @patch('sys.argv', ['cli.py', '--rag-search'])
    def test_missing_rag_search_query_error(self):
        """Test error when --rag-search has no value."""
        with pytest.raises(SystemExit):
            with patch('sys.stderr', new=StringIO()):
                main()


class TestCLIBasicResearch:
    """Test basic CLI research functionality."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'lung cancer detection'])
    def test_basic_research_query(self, mock_validate, mock_agent_class):
        """Test basic research query."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {'summary': 'Test summary'}
        mock_agent.format_results.return_value = "Formatted results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            mock_agent.research.assert_called_once_with('lung cancer detection')
            assert 'Formatted results' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test query', '--display-results', '10'])
    def test_research_with_display_limit(self, mock_validate, mock_agent_class):
        """Test research with custom display limit."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            mock_agent.format_results.assert_called_once_with({}, display_limit=10)
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--max-results', '50'])
    def test_research_with_max_results(self, mock_validate, mock_agent_class):
        """Test research with custom max results."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['max_results'] == 50


class TestCLILLMProvider:
    """Test CLI LLM provider options."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--llm-provider', 'openai'])
    def test_openai_provider(self, mock_validate, mock_agent_class):
        """Test using OpenAI provider."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'openai'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--llm-provider', 'github'])
    def test_github_provider(self, mock_validate, mock_agent_class):
        """Test using GitHub provider."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'github'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--llm-provider', 'none'])
    def test_none_provider(self, mock_validate, mock_agent_class):
        """Test using none provider."""
        mock_validate.return_value = (True, "None provider")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'none'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--no-summary'])
    def test_no_summary_flag(self, mock_validate, mock_agent_class):
        """Test --no-summary flag sets provider to none."""
        mock_validate.return_value = (True, "None")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'none'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--llm-provider', 'openai'])
    def test_provider_fallback_on_invalid_config(self, mock_validate, mock_agent_class):
        """Test fallback to none when provider not configured."""
        mock_validate.return_value = (False, "API key missing")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert '⚠' in output
            assert 'Falling back to no-summary mode' in output
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'none'


class TestCLIModelOptions:
    """Test CLI model and temperature options."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--model', 'gpt-4o'])
    def test_custom_model(self, mock_validate, mock_agent_class):
        """Test using custom model."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['model_name'] == 'gpt-4o'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--temperature', '0.5'])
    def test_custom_temperature(self, mock_validate, mock_agent_class):
        """Test using custom temperature."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['temperature'] == 0.5


class TestCLIRAGFunctionality:
    """Test CLI RAG functionality."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--no-rag'])
    def test_disable_rag(self, mock_validate, mock_agent_class):
        """Test disabling RAG functionality."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['enable_rag'] is False
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test'])
    def test_rag_enabled_by_default(self, mock_validate, mock_agent_class):
        """Test RAG is enabled by default."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['enable_rag'] is True
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--rag-dir', '/custom/path'])
    def test_custom_rag_directory(self, mock_validate, mock_agent_class):
        """Test custom RAG directory."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['rag_persist_dir'] == '/custom/path'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-stats'])
    def test_rag_stats_success(self, mock_validate, mock_agent_class):
        """Test RAG stats command."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.get_rag_stats.return_value = {
            'collection_name': 'test_collection',
            'document_count': 42,
            'persist_directory': './chroma_db'
        }
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'RAG Database Statistics:' in output
            assert 'test_collection' in output
            assert '42' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-stats'])
    def test_rag_stats_error(self, mock_validate, mock_agent_class):
        """Test RAG stats with error."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.get_rag_stats.return_value = {
            'error': 'RAG not enabled'
        }
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            assert 'Error: RAG not enabled' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-search', 'deep learning'])
    def test_rag_search_basic(self, mock_validate, mock_agent_class):
        """Test basic RAG search."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.search_rag.return_value = [{'content': 'test'}]
        mock_agent.format_rag_results.return_value = "RAG results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            mock_agent.search_rag.assert_called_once_with(
                query='deep learning',
                k=10,
                source_filter=None
            )
            assert 'RAG results' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', '--rag-search', 'test', '--rag-source', 'arxiv', '--rag-results', '20'])
    def test_rag_search_with_filters(self, mock_validate, mock_agent_class):
        """Test RAG search with source filter and custom results count."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.search_rag.return_value = []
        mock_agent.format_rag_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            mock_agent.search_rag.assert_called_once_with(
                query='test',
                k=20,
                source_filter='arxiv'
            )
            mock_agent.format_rag_results.assert_called_once_with([], show_content=True)


class TestCLIOutputFile:
    """Test CLI output file functionality."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('sys.argv', ['cli.py', 'test', '--output', 'results.txt'])
    def test_save_to_file(self, mock_file, mock_validate, mock_agent_class):
        """Test saving results to file."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Formatted output"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            mock_file.assert_called_once_with('results.txt', 'w', encoding='utf-8')
            handle = mock_file()
            handle.write.assert_called_once_with("Formatted output")
            assert 'Results saved to results.txt' in output


class TestCLIPubMedEmail:
    """Test CLI PubMed email option."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test', '--email', 'user@example.com'])
    def test_custom_email(self, mock_validate, mock_agent_class):
        """Test custom PubMed email."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['pubmed_email'] == 'user@example.com'
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test'])
    def test_default_email(self, mock_validate, mock_agent_class):
        """Test default PubMed email."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {}
        mock_agent.format_results.return_value = "Results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()):
            main()
            
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['pubmed_email'] == 'research@example.com'


class TestCLIExceptionHandling:
    """Test CLI exception handling."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test'])
    def test_agent_initialization_error(self, mock_validate, mock_agent_class):
        """Test handling of agent initialization error."""
        mock_validate.return_value = (True, "Configured")
        mock_agent_class.side_effect = Exception("Initialization failed")
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                main()
                assert 'Initialization failed' in mock_stderr.getvalue()
        
        assert exc_info.value.code == 1
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', ['cli.py', 'test'])
    def test_research_error(self, mock_validate, mock_agent_class):
        """Test handling of research error."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.side_effect = Exception("Research failed")
        mock_agent_class.return_value = mock_agent
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                main()
                assert 'Research failed' in mock_stderr.getvalue()
        
        assert exc_info.value.code == 1


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('sys.argv', [
        'cli.py', 'lung cancer',
        '--llm-provider', 'github',
        '--model', 'gpt-4o',
        '--temperature', '0.8',
        '--max-results', '30',
        '--display-results', '15',
        '--email', 'test@test.com'
    ])
    def test_full_research_workflow(self, mock_validate, mock_agent_class):
        """Test complete research workflow with all options."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.research.return_value = {
            'summary': 'Test summary',
            'arxiv_results': [],
            'pubmed_results': [],
            'huggingface_results': []
        }
        mock_agent.format_results.return_value = "Complete results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            # Verify agent initialization
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs['llm_provider'] == 'github'
            assert call_kwargs['model_name'] == 'gpt-4o'
            assert call_kwargs['temperature'] == 0.8
            assert call_kwargs['max_results'] == 30
            assert call_kwargs['pubmed_email'] == 'test@test.com'
            
            # Verify research call
            mock_agent.research.assert_called_once_with('lung cancer')
            
            # Verify formatting
            mock_agent.format_results.assert_called_once()
            assert 'Complete results' in output
    
    @patch('research_viz_agent.cli.MedicalCVResearchAgent')
    @patch('research_viz_agent.cli.LLMFactory.validate_provider_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('sys.argv', [
        'cli.py', '--rag-search', 'medical AI',
        '--rag-source', 'pubmed',
        '--rag-results', '25',
        '--output', 'rag_output.txt'
    ])
    def test_rag_search_with_output(self, mock_file, mock_validate, mock_agent_class):
        """Test RAG search with file output."""
        mock_validate.return_value = (True, "Configured")
        
        mock_agent = MagicMock()
        mock_agent.search_rag.return_value = [
            {'content': 'Result 1'},
            {'content': 'Result 2'}
        ]
        mock_agent.format_rag_results.return_value = "RAG search results"
        mock_agent_class.return_value = mock_agent
        
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            
            # Verify RAG search
            mock_agent.search_rag.assert_called_once_with(
                query='medical AI',
                k=25,
                source_filter='pubmed'
            )
            
            # Verify file save
            mock_file.assert_called_once_with('rag_output.txt', 'w', encoding='utf-8')
            handle = mock_file()
            handle.write.assert_called_once_with("RAG search results")
            
            assert 'Results saved to rag_output.txt' in output
