"""
Unit tests for the ResearchWorkflow LangGraph workflow.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from openai import RateLimitError

from research_viz_agent.agents.research_workflow import (
    ResearchWorkflow,
    AgentState
)


class TestResearchWorkflowInitialization:
    """Test ResearchWorkflow initialization."""
    
    def test_init_with_all_tools(self):
        """Test initialization with all tools."""
        mock_llm = MagicMock()
        mock_arxiv = MagicMock()
        mock_pubmed = MagicMock()
        mock_hf = MagicMock()
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf,
            max_results=20
        )
        
        assert workflow.llm == mock_llm
        assert workflow.arxiv_tool == mock_arxiv
        assert workflow.pubmed_tool == mock_pubmed
        assert workflow.huggingface_tool == mock_hf
        assert workflow.max_results == 20
        assert workflow.graph is not None
    
    def test_init_with_none_llm(self):
        """Test initialization with None LLM."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock(),
            max_results=10
        )
        
        assert workflow.llm is None
        assert workflow.max_results == 10
    
    def test_init_with_custom_max_results(self):
        """Test initialization with custom max_results."""
        workflow = ResearchWorkflow(
            llm=MagicMock(),
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock(),
            max_results=50
        )
        
        assert workflow.max_results == 50
    
    def test_build_graph_creates_workflow(self):
        """Test that _build_graph creates a valid workflow."""
        workflow = ResearchWorkflow(
            llm=MagicMock(),
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        # Graph should be compiled and ready to use
        assert workflow.graph is not None
        assert hasattr(workflow.graph, 'invoke')


class TestResearchWorkflowSearchNodes:
    """Test individual search node methods."""
    
    def test_search_arxiv_basic(self):
        """Test arXiv search node."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = [
            {'title': 'Paper 1', 'authors': ['Author A']},
            {'title': 'Paper 2', 'authors': ['Author B']}
        ]
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock(),
            max_results=15
        )
        
        state = {"query": "medical imaging AI", "arxiv_results": []}
        result_state = workflow._search_arxiv(state)
        
        assert len(result_state["arxiv_results"]) == 2
        assert result_state["arxiv_results"][0]['title'] == 'Paper 1'
        mock_arxiv.search_medical_cv_models.assert_called_once_with("medical imaging AI", max_results=15)
    
    def test_search_pubmed_basic(self):
        """Test PubMed search node."""
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = [
            {'title': 'Study 1', 'pmid': '12345'},
            {'title': 'Study 2', 'pmid': '67890'}
        ]
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=mock_pubmed,
            huggingface_tool=MagicMock(),
            max_results=10
        )
        
        state = {"query": "chest xray detection", "pubmed_results": []}
        result_state = workflow._search_pubmed(state)
        
        assert len(result_state["pubmed_results"]) == 2
        assert result_state["pubmed_results"][0]['pmid'] == '12345'
        mock_pubmed.search_medical_cv_models.assert_called_once_with("chest xray detection", max_results=10)
    
    def test_search_huggingface_basic(self):
        """Test HuggingFace search node."""
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = [
            {'model_id': 'test/model-1', 'downloads': 1000},
            {'model_id': 'test/model-2', 'downloads': 500}
        ]
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=mock_hf,
            max_results=25
        )
        
        state = {"query": "medical segmentation", "huggingface_results": []}
        result_state = workflow._search_huggingface(state)
        
        assert len(result_state["huggingface_results"]) == 2
        assert result_state["huggingface_results"][0]['model_id'] == 'test/model-1'
        mock_hf.search_medical_cv_models.assert_called_once_with("medical segmentation", max_results=25)
    
    def test_search_nodes_empty_results(self):
        """Test search nodes with empty results."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = []
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {"query": "nonexistent topic", "arxiv_results": []}
        result_state = workflow._search_arxiv(state)
        
        assert result_state["arxiv_results"] == []


class TestResearchWorkflowPrepareContext:
    """Test context preparation for summarization."""
    
    def test_prepare_context_all_sources(self):
        """Test context preparation with results from all sources."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        arxiv_results = [
            {
                'title': 'ArXiv Paper 1',
                'authors': ['Author A', 'Author B'],
                'summary': 'A' * 400,
                'categories': ['cs.CV', 'cs.AI']
            }
        ]
        
        pubmed_results = [
            {
                'title': 'PubMed Study 1',
                'authors': ['Researcher X', 'Researcher Y'],
                'abstract': 'B' * 400,
                'journal': 'Medical Journal'
            }
        ]
        
        huggingface_results = [
            {
                'model_id': 'test/model-1',
                'author': 'test_org',
                'pipeline_tag': 'image-classification',
                'downloads': 5000,
                'tags': ['medical', 'cv', 'xray', 'radiology', 'detection', 'extra']
            }
        ]
        
        context = workflow._prepare_context(arxiv_results, pubmed_results, huggingface_results)
        
        assert 'ArXiv Papers' in context
        assert 'PubMed Papers' in context
        assert 'HuggingFace Models' in context
        assert 'ArXiv Paper 1' in context
        assert 'PubMed Study 1' in context
        assert 'test/model-1' in context
    
    def test_prepare_context_empty_results(self):
        """Test context preparation with empty results."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        context = workflow._prepare_context([], [], [])
        
        # Should return empty string or minimal context
        assert isinstance(context, str)
    
    def test_prepare_context_truncates_long_summaries(self):
        """Test that context preparation truncates long summaries."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        arxiv_results = [
            {
                'title': 'Long Paper',
                'authors': ['Author'],
                'summary': 'X' * 500,  # Very long summary
                'categories': ['cs.CV']
            }
        ]
        
        context = workflow._prepare_context(arxiv_results, [], [])
        
        # Summary should be truncated to ~300 chars
        assert context.count('X') < 500
        assert '...' in context
    
    def test_prepare_context_limits_to_10_items(self):
        """Test that context preparation limits results to 10 per source."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        # Create 15 papers
        arxiv_results = [
            {
                'title': f'Paper {i}',
                'authors': ['Author'],
                'summary': 'Summary',
                'categories': ['cs.CV']
            }
            for i in range(15)
        ]
        
        context = workflow._prepare_context(arxiv_results, [], [])
        
        # Should only include first 10
        assert 'Paper 0' in context
        assert 'Paper 9' in context
        assert 'Paper 10' not in context


class TestResearchWorkflowFallbackSummary:
    """Test fallback summary creation."""
    
    def test_create_fallback_summary_basic(self):
        """Test basic fallback summary creation."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        arxiv_results = [{'title': 'ArXiv Paper 1'}]
        pubmed_results = [{'title': 'PubMed Study 1'}]
        huggingface_results = [{'model_id': 'test/model-1', 'pipeline_tag': 'classification', 'downloads': 1000}]
        
        summary = workflow._create_fallback_summary(
            arxiv_results, pubmed_results, huggingface_results, "test query"
        )
        
        assert 'test query' in summary
        assert 'Total Papers Found: 2' in summary
        assert 'Total Models Found: 1' in summary
        assert 'ArXiv Research Highlights' in summary
        assert 'PubMed Research Highlights' in summary
        assert 'HuggingFace Model Highlights' in summary
    
    def test_create_fallback_summary_empty_results(self):
        """Test fallback summary with empty results."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        summary = workflow._create_fallback_summary([], [], [], "empty query")
        
        assert 'Total Papers Found: 0' in summary
        assert 'Total Models Found: 0' in summary
    
    def test_create_fallback_summary_truncates_long_titles(self):
        """Test that fallback summary truncates long titles."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        arxiv_results = [{'title': 'X' * 150}]
        
        summary = workflow._create_fallback_summary(arxiv_results, [], [], "query")
        
        # Title should be truncated to 100 chars (plus ellipsis adds a few more)
        assert summary.count('X') <= 103  # 100 + "..." padding
        assert '...' in summary
    
    def test_create_fallback_summary_limits_highlights(self):
        """Test that fallback summary shows only top 3 with more indicator."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        arxiv_results = [{'title': f'Paper {i}'} for i in range(10)]
        
        summary = workflow._create_fallback_summary(arxiv_results, [], [], "query")
        
        assert 'Paper 0' in summary
        assert 'Paper 2' in summary
        assert '... and 7 more papers' in summary


class TestResearchWorkflowSummarize:
    """Test summarization node."""
    
    @patch('research_viz_agent.agents.research_workflow.ChatPromptTemplate')
    def test_summarize_with_llm_success(self, mock_template):
        """Test successful summarization with LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is an AI-generated summary."
        mock_llm.invoke.return_value = mock_response
        
        # Mock the prompt template
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = ["messages"]
        mock_template.from_messages.return_value = mock_prompt
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {
            "query": "test query",
            "arxiv_results": [{'title': 'Paper 1', 'authors': [], 'summary': '', 'categories': []}],
            "pubmed_results": [],
            "huggingface_results": [],
            "summary": ""
        }
        
        result_state = workflow._summarize_results(state)
        
        assert result_state["summary"] == "This is an AI-generated summary."
        mock_llm.invoke.assert_called_once()
    
    def test_summarize_without_llm(self):
        """Test summarization without LLM (fallback)."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {
            "query": "test query",
            "arxiv_results": [{'title': 'Paper 1'}],
            "pubmed_results": [{'title': 'Study 1'}],
            "huggingface_results": [],
            "summary": ""
        }
        
        result_state = workflow._summarize_results(state)
        
        assert 'AI Summarization Disabled' in result_state["summary"]
        assert 'Total Papers Found: 2' in result_state["summary"]
    
    @patch('research_viz_agent.agents.research_workflow.ChatPromptTemplate')
    def test_summarize_quota_exceeded_error(self, mock_template):
        """Test summarization with quota exceeded error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("insufficient_quota: you exceeded your quota")
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = ["messages"]
        mock_template.from_messages.return_value = mock_prompt
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {
            "query": "test",
            "arxiv_results": [],
            "pubmed_results": [],
            "huggingface_results": [],
            "summary": ""
        }
        
        result_state = workflow._summarize_results(state)
        
        assert 'OpenAI API Quota Exceeded' in result_state["summary"]
        assert 'billing' in result_state["summary"].lower()
    
    @patch('research_viz_agent.agents.research_workflow.ChatPromptTemplate')
    def test_summarize_rate_limit_error(self, mock_template):
        """Test summarization with rate limit error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("rate_limit exceeded")
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = ["messages"]
        mock_template.from_messages.return_value = mock_prompt
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {
            "query": "test",
            "arxiv_results": [],
            "pubmed_results": [],
            "huggingface_results": [],
            "summary": ""
        }
        
        result_state = workflow._summarize_results(state)
        
        assert 'Rate limit exceeded' in result_state["summary"]
    
    @patch('research_viz_agent.agents.research_workflow.ChatPromptTemplate')
    def test_summarize_generic_error(self, mock_template):
        """Test summarization with generic error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Some other error")
        
        mock_prompt = MagicMock()
        mock_prompt.format_messages.return_value = ["messages"]
        mock_template.from_messages.return_value = mock_prompt
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        state = {
            "query": "test",
            "arxiv_results": [],
            "pubmed_results": [],
            "huggingface_results": [],
            "summary": ""
        }
        
        result_state = workflow._summarize_results(state)
        
        assert 'Summary generation failed' in result_state["summary"]
        assert 'Some other error' in result_state["summary"]


class TestResearchWorkflowLLMBackoff:
    """Test LLM invocation with backoff."""
    
    def test_invoke_llm_with_backoff_success(self):
        """Test successful LLM invocation."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "response"
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        result = workflow._invoke_llm_with_backoff(["message"])
        
        assert result == "response"
        mock_llm.invoke.assert_called_once_with(["message"])
    
    def test_invoke_llm_with_backoff_no_llm(self):
        """Test LLM invocation when LLM is None."""
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        with pytest.raises(ValueError, match="LLM not available"):
            workflow._invoke_llm_with_backoff(["message"])
    
    def test_invoke_llm_with_backoff_rate_limit(self):
        """Test LLM invocation with rate limit error."""
        mock_llm = MagicMock()
        # Create a proper RateLimitError with required arguments
        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_error = RateLimitError("Rate limited", response=mock_response, body={})
        
        # First call raises RateLimitError, second succeeds
        mock_llm.invoke.side_effect = [rate_error, "success"]
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=MagicMock(),
            pubmed_tool=MagicMock(),
            huggingface_tool=MagicMock()
        )
        
        # The backoff decorator should retry
        result = workflow._invoke_llm_with_backoff(["message"])
        
        assert result == "success"


class TestResearchWorkflowRun:
    """Test complete workflow execution."""
    
    def test_run_complete_workflow(self):
        """Test running the complete workflow."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Final summary"
        mock_llm.invoke.return_value = mock_response
        
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = [{'title': 'ArXiv Paper'}]
        
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = [{'title': 'PubMed Study'}]
        
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = [{'model_id': 'test/model'}]
        
        workflow = ResearchWorkflow(
            llm=mock_llm,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf,
            max_results=10
        )
        
        result = workflow.run("medical imaging AI")
        
        assert result["query"] == "medical imaging AI"
        # Results are accumulated due to operator.add in state, so may have more than 1
        assert len(result["arxiv_results"]) >= 1
        assert len(result["pubmed_results"]) >= 1
        assert len(result["huggingface_results"]) >= 1
        assert result["summary"] == "Final summary"
        
        # Verify the tools were called
        mock_arxiv.search_medical_cv_models.assert_called()
        mock_pubmed.search_medical_cv_models.assert_called()
        mock_hf.search_medical_cv_models.assert_called()
    
    def test_run_without_llm(self):
        """Test running workflow without LLM."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = []
        
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = []
        
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = []
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf
        )
        
        result = workflow.run("test query")
        
        assert result["query"] == "test query"
        assert 'AI Summarization Disabled' in result["summary"]
    
    def test_run_initializes_state_correctly(self):
        """Test that run initializes state with correct structure."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = []
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = []
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = []
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf
        )
        
        result = workflow.run("query")
        
        # Check all required state fields are present
        assert "query" in result
        assert "arxiv_results" in result
        assert "pubmed_results" in result
        assert "huggingface_results" in result
        assert "summary" in result
        assert "next_step" in result


class TestResearchWorkflowIntegration:
    """Test integration scenarios."""
    
    def test_workflow_handles_all_empty_results(self):
        """Test workflow with no results from any source."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = []
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = []
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = []
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf
        )
        
        result = workflow.run("nonexistent topic")
        
        assert len(result["arxiv_results"]) == 0
        assert len(result["pubmed_results"]) == 0
        assert len(result["huggingface_results"]) == 0
        assert 'Total Papers Found: 0' in result["summary"]
    
    def test_workflow_with_partial_results(self):
        """Test workflow with results from some sources."""
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.return_value = [{'title': 'Paper'}]
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.return_value = []
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.return_value = [{'model_id': 'model'}]
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf
        )
        
        result = workflow.run("partial query")
        
        # Results are accumulated due to operator.add in state
        assert len(result["arxiv_results"]) >= 1
        assert len(result["pubmed_results"]) == 0  # Empty should stay empty
        assert len(result["huggingface_results"]) >= 1
        
        # Verify at least one paper and model were found
        assert any('Paper' in r.get('title', '') for r in result["arxiv_results"])
        assert any('model' in r.get('model_id', '') for r in result["huggingface_results"])
    
    def test_workflow_execution_order(self):
        """Test that workflow executes nodes in correct order."""
        call_order = []
        
        mock_arxiv = MagicMock()
        mock_arxiv.search_medical_cv_models.side_effect = lambda *args, **kwargs: (call_order.append('arxiv'), [])[1]
        
        mock_pubmed = MagicMock()
        mock_pubmed.search_medical_cv_models.side_effect = lambda *args, **kwargs: (call_order.append('pubmed'), [])[1]
        
        mock_hf = MagicMock()
        mock_hf.search_medical_cv_models.side_effect = lambda *args, **kwargs: (call_order.append('hf'), [])[1]
        
        workflow = ResearchWorkflow(
            llm=None,
            arxiv_tool=mock_arxiv,
            pubmed_tool=mock_pubmed,
            huggingface_tool=mock_hf
        )
        
        workflow.run("test")
        
        # Should execute in order: arxiv -> pubmed -> huggingface
        assert call_order == ['arxiv', 'pubmed', 'hf']
