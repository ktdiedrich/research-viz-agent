"""
Unit tests for the HuggingFace MCP tool.
"""
from unittest.mock import MagicMock, patch, Mock
import mcp.types as types

from research_viz_agent.mcp_tools.huggingface_tool import (
    HuggingFaceTool,
    create_huggingface_tool,
    get_huggingface_server,
    search_huggingface_models,
    search_medical_cv_models,
    get_model_details
)


class TestHuggingFaceToolInitialization:
    """Test HuggingFaceTool initialization."""
    
    def test_init_without_token(self):
        """Test initialization without token."""
        with patch.dict('os.environ', {}, clear=True):
            tool = HuggingFaceTool()
            assert tool.token is None
            assert tool.base_url == "https://huggingface.co/api"
            assert tool.headers == {}
    
    def test_init_with_token_parameter(self):
        """Test initialization with token parameter."""
        tool = HuggingFaceTool(token="test_token_123")
        assert tool.token == "test_token_123"
        assert tool.headers == {"Authorization": "Bearer test_token_123"}
    
    def test_init_with_env_token(self):
        """Test initialization with environment variable token."""
        with patch.dict('os.environ', {'HUGGINGFACE_TOKEN': 'env_token_456'}):
            tool = HuggingFaceTool()
            assert tool.token == "env_token_456"
            assert tool.headers == {"Authorization": "Bearer env_token_456"}
    
    def test_create_huggingface_tool_factory(self):
        """Test factory function creates HuggingFaceTool instance."""
        tool = create_huggingface_tool(token="factory_token")
        assert isinstance(tool, HuggingFaceTool)
        assert tool.token == "factory_token"
    
    def test_get_huggingface_server(self):
        """Test server instance retrieval."""
        from mcp.server import Server
        server = get_huggingface_server()
        assert isinstance(server, Server)


class TestHuggingFaceToolSearchModels:
    """Test search_models functionality."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_basic(self, mock_get):
        """Test basic model search."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                'id': 'test/model-1',
                'author': 'test',
                'modelId': 'test/model-1',
                'downloads': 1000,
                'likes': 50,
                'tags': ['computer-vision', 'medical'],
                'pipeline_tag': 'image-classification',
                'createdAt': '2024-01-01T00:00:00Z',
                'lastModified': '2024-01-02T00:00:00Z',
                'library_name': 'transformers'
            }
        ]
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_models(search_query="medical", limit=10)
        
        assert len(results) == 1
        assert results[0]['model_id'] == 'test/model-1'
        assert results[0]['author'] == 'test'
        assert results[0]['downloads'] == 1000
        assert results[0]['likes'] == 50
        assert 'medical' in results[0]['tags']
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'search' in call_args[1]['params']
        assert call_args[1]['params']['search'] == 'medical'
        assert call_args[1]['params']['limit'] == 10
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_with_task_filter(self, mock_get):
        """Test search with task filter."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_models(search_query="test", task="object-detection", limit=5)
        
        call_args = mock_get.call_args
        assert call_args[1]['params']['filter'] == 'object-detection'
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_with_tags(self, mock_get):
        """Test search with tags filter."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_models(search_query="test", tags=["medical", "cv"], limit=5)
        
        call_args = mock_get.call_args
        assert call_args[1]['params']['tags'] == 'medical,cv'
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_empty_results(self, mock_get):
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_models(search_query="nonexistent", limit=10)
        
        assert len(results) == 0
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_multiple_results(self, mock_get):
        """Test search returning multiple models."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                'id': f'test/model-{i}',
                'author': 'test',
                'modelId': f'test/model-{i}',
                'downloads': 100 * i,
                'likes': 10 * i,
                'tags': ['tag1', 'tag2'],
                'pipeline_tag': 'image-classification',
                'createdAt': '2024-01-01T00:00:00Z',
                'lastModified': '2024-01-02T00:00:00Z',
                'library_name': 'transformers'
            }
            for i in range(1, 6)
        ]
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_models(limit=5)
        
        assert len(results) == 5
        assert results[0]['model_id'] == 'test/model-1'
        assert results[4]['model_id'] == 'test/model-5'
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_with_auth_token(self, mock_get):
        """Test search with authentication token."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool(token="auth_token")
        tool.search_models(limit=5)
        
        call_args = mock_get.call_args
        assert call_args[1]['headers'] == {"Authorization": "Bearer auth_token"}
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_error_handling(self, mock_get):
        """Test error handling in search."""
        mock_get.side_effect = Exception("API Error")
        
        tool = HuggingFaceTool()
        results = tool.search_models(search_query="test")
        
        assert results == []
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_models(search_query="test")
        
        assert results == []
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_models_missing_fields(self, mock_get):
        """Test handling of models with missing fields."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                'id': 'test/model-1',
                # Missing many fields
            }
        ]
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_models(limit=1)
        
        assert len(results) == 1
        assert results[0]['model_id'] == 'test/model-1'
        assert results[0]['author'] == ''
        assert results[0]['downloads'] == 0
        assert results[0]['likes'] == 0
        assert results[0]['tags'] == []


class TestHuggingFaceToolSearchMedicalCV:
    """Test search_medical_cv_models functionality."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_medical_cv_basic(self, mock_get):
        """Test basic medical CV model search."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                'id': 'medical/xray-model',
                'author': 'medical',
                'modelId': 'medical/xray-model',
                'downloads': 5000,
                'likes': 100,
                'tags': ['medical', 'xray', 'image-classification'],
                'pipeline_tag': 'image-classification',
                'createdAt': '2024-01-01T00:00:00Z',
                'lastModified': '2024-01-02T00:00:00Z',
                'library_name': 'transformers'
            }
        ]
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        results = tool.search_medical_cv_models()
        
        assert len(results) == 1
        assert results[0]['model_id'] == 'medical/xray-model'
        
        # Verify the search query includes medical imaging
        call_args = mock_get.call_args
        assert 'medical imaging' in call_args[1]['params']['search']
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_medical_cv_with_additional_query(self, mock_get):
        """Test medical CV search with additional terms."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_medical_cv_models(additional_query="chest xray", max_results=15)
        
        call_args = mock_get.call_args
        assert 'medical imaging chest xray' in call_args[1]['params']['search']
        assert call_args[1]['params']['limit'] == 15
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_medical_cv_task_filter(self, mock_get):
        """Test that medical CV search filters by image-classification task."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_medical_cv_models()
        
        call_args = mock_get.call_args
        assert call_args[1]['params']['filter'] == 'image-classification'
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_search_medical_cv_custom_max_results(self, mock_get):
        """Test medical CV search with custom max results."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_medical_cv_models(max_results=30)
        
        call_args = mock_get.call_args
        assert call_args[1]['params']['limit'] == 30


class TestHuggingFaceToolGetModelInfo:
    """Test get_model_info functionality."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_get_model_info_success(self, mock_get):
        """Test successful model info retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'id': 'microsoft/resnet-50',
            'author': 'microsoft',
            'downloads': 100000,
            'likes': 500,
            'tags': ['computer-vision', 'image-classification'],
            'pipeline_tag': 'image-classification',
            'library_name': 'transformers'
        }
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        info = tool.get_model_info('microsoft/resnet-50')
        
        assert info is not None
        assert info['id'] == 'microsoft/resnet-50'
        assert info['author'] == 'microsoft'
        assert info['downloads'] == 100000
        
        # Verify API call
        mock_get.assert_called_once()
        assert 'microsoft/resnet-50' in mock_get.call_args[0][0]
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_get_model_info_error(self, mock_get):
        """Test error handling in get_model_info."""
        mock_get.side_effect = Exception("Model not found")
        
        tool = HuggingFaceTool()
        info = tool.get_model_info('nonexistent/model')
        
        assert info is None
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_get_model_info_http_error(self, mock_get):
        """Test HTTP error in get_model_info."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        info = tool.get_model_info('invalid/model')
        
        assert info is None
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_get_model_info_with_auth(self, mock_get):
        """Test model info retrieval with authentication."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'id': 'test/model'}
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool(token="auth_token")
        tool.get_model_info('test/model')
        
        call_args = mock_get.call_args
        assert call_args[1]['headers'] == {"Authorization": "Bearer auth_token"}


class TestSearchHuggingFaceModelsAsync:
    """Test async search_huggingface_models MCP function."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_success(self, mock_tool):
        """Test successful async model search."""
        mock_tool.search_models.return_value = [
            {
                'model_id': 'test/model-1',
                'author': 'test',
                'pipeline_tag': 'image-classification',
                'downloads': 1000,
                'likes': 50,
                'tags': ['cv', 'medical'],
                'library_name': 'transformers',
                'model_card_url': 'https://huggingface.co/test/model-1'
            }
        ]
        
        result = await search_huggingface_models({
            'search_query': 'medical',
            'limit': 10
        })
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert 'test/model-1' in result[0].text
        assert 'Found 1 HuggingFace models' in result[0].text
        
        mock_tool.search_models.assert_called_once_with(
            search_query='medical',
            task=None,
            tags=None,
            limit=10
        )
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_with_filters(self, mock_tool):
        """Test async search with task and tags filters."""
        mock_tool.search_models.return_value = []
        
        await search_huggingface_models({
            'search_query': 'medical',
            'task': 'object-detection',
            'tags': 'medical, xray, radiology',
            'limit': 20
        })
        
        mock_tool.search_models.assert_called_once_with(
            search_query='medical',
            task='object-detection',
            tags=['medical', 'xray', 'radiology'],
            limit=20
        )
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_no_results(self, mock_tool):
        """Test async search with no results."""
        mock_tool.search_models.return_value = []
        
        result = await search_huggingface_models({'search_query': 'nonexistent'})
        
        assert len(result) == 1
        assert 'No models found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_error_handling(self, mock_tool):
        """Test error handling in async search."""
        mock_tool.search_models.side_effect = Exception("API Error")
        
        result = await search_huggingface_models({'search_query': 'test'})
        
        assert len(result) == 1
        assert 'Error searching HuggingFace models' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_default_params(self, mock_tool):
        """Test async search with default parameters."""
        mock_tool.search_models.return_value = []
        
        await search_huggingface_models({})
        
        mock_tool.search_models.assert_called_once_with(
            search_query='',
            task=None,
            tags=None,
            limit=20
        )
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_huggingface_models_formatting(self, mock_tool):
        """Test result formatting with multiple models."""
        mock_tool.search_models.return_value = [
            {
                'model_id': f'test/model-{i}',
                'author': 'test',
                'pipeline_tag': 'image-classification',
                'downloads': 1000 * i,
                'likes': 50 * i,
                'tags': ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6'],
                'library_name': 'transformers',
                'model_card_url': f'https://huggingface.co/test/model-{i}'
            }
            for i in range(1, 4)
        ]
        
        result = await search_huggingface_models({'limit': 3})
        
        assert len(result) == 1
        text = result[0].text
        assert 'Found 3 HuggingFace models' in text
        assert 'test/model-1' in text
        assert 'test/model-3' in text
        assert '1,000' in text  # Check formatting
        assert '...' in text  # Check tag truncation


class TestSearchMedicalCVModelsAsync:
    """Test async search_medical_cv_models MCP function."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_medical_cv_models_success(self, mock_tool):
        """Test successful async medical CV model search."""
        mock_tool.search_medical_cv_models.return_value = [
            {
                'model_id': 'medical/xray-classifier',
                'author': 'medical',
                'pipeline_tag': 'image-classification',
                'downloads': 5000,
                'likes': 200,
                'tags': ['medical', 'xray'],
                'library_name': 'transformers',
                'model_card_url': 'https://huggingface.co/medical/xray-classifier'
            }
        ]
        
        result = await search_medical_cv_models({
            'additional_query': 'chest xray',
            'max_results': 15
        })
        
        assert len(result) == 1
        assert 'Found 1 medical CV models' in result[0].text
        assert 'medical/xray-classifier' in result[0].text
        
        mock_tool.search_medical_cv_models.assert_called_once_with('chest xray', 15)
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_medical_cv_models_no_results(self, mock_tool):
        """Test async medical CV search with no results."""
        mock_tool.search_medical_cv_models.return_value = []
        
        result = await search_medical_cv_models({})
        
        assert len(result) == 1
        assert 'No medical computer vision models found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_medical_cv_models_default_params(self, mock_tool):
        """Test async medical CV search with default parameters."""
        mock_tool.search_medical_cv_models.return_value = []
        
        await search_medical_cv_models({})
        
        mock_tool.search_medical_cv_models.assert_called_once_with('', 20)
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_search_medical_cv_models_error_handling(self, mock_tool):
        """Test error handling in async medical CV search."""
        mock_tool.search_medical_cv_models.side_effect = Exception("Search failed")
        
        result = await search_medical_cv_models({})
        
        assert len(result) == 1
        assert 'Error searching medical CV models' in result[0].text


class TestGetModelDetailsAsync:
    """Test async get_model_details MCP function."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_get_model_details_success(self, mock_tool):
        """Test successful async model details retrieval."""
        mock_tool.get_model_info.return_value = {
            'id': 'microsoft/resnet-50',
            'author': 'microsoft',
            'pipeline_tag': 'image-classification',
            'downloads': 100000,
            'likes': 500,
            'createdAt': '2023-01-01T00:00:00Z',
            'lastModified': '2024-01-01T00:00:00Z',
            'library_name': 'transformers',
            'tags': ['computer-vision', 'image-classification']
        }
        
        result = await get_model_details({'model_id': 'microsoft/resnet-50'})
        
        assert len(result) == 1
        text = result[0].text
        assert 'microsoft/resnet-50' in text
        assert 'microsoft' in text
        assert '100,000' in text
        assert 'transformers' in text
        
        mock_tool.get_model_info.assert_called_once_with('microsoft/resnet-50')
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_get_model_details_missing_model_id(self, mock_tool):
        """Test async model details with missing model_id."""
        result = await get_model_details({})
        
        assert len(result) == 1
        assert 'Error: model_id parameter is required' in result[0].text
        mock_tool.get_model_info.assert_not_called()
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_get_model_details_model_not_found(self, mock_tool):
        """Test async model details when model not found."""
        mock_tool.get_model_info.return_value = None
        
        result = await get_model_details({'model_id': 'nonexistent/model'})
        
        assert len(result) == 1
        assert 'not found' in result[0].text
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.huggingface_tool')
    async def test_get_model_details_error_handling(self, mock_tool):
        """Test error handling in async model details."""
        mock_tool.get_model_info.side_effect = Exception("API Error")
        
        result = await get_model_details({'model_id': 'test/model'})
        
        assert len(result) == 1
        assert 'Error getting model details' in result[0].text


class TestHuggingFaceToolIntegration:
    """Test integration scenarios."""
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_full_workflow_search_to_details(self, mock_get):
        """Test complete workflow from search to getting model details."""
        # Mock search response
        search_response = MagicMock()
        search_response.json.return_value = [
            {
                'id': 'test/medical-model',
                'author': 'test',
                'modelId': 'test/medical-model',
                'downloads': 1000,
                'likes': 50,
                'tags': ['medical'],
                'pipeline_tag': 'image-classification',
                'createdAt': '2024-01-01T00:00:00Z',
                'lastModified': '2024-01-02T00:00:00Z',
                'library_name': 'transformers'
            }
        ]
        
        # Mock details response
        details_response = MagicMock()
        details_response.json.return_value = {
            'id': 'test/medical-model',
            'author': 'test',
            'downloads': 1000
        }
        
        mock_get.side_effect = [search_response, details_response]
        
        tool = HuggingFaceTool()
        
        # First search
        results = tool.search_models(search_query="medical", limit=1)
        assert len(results) == 1
        model_id = results[0]['model_id']
        
        # Then get details
        details = tool.get_model_info(model_id)
        assert details is not None
        assert details['id'] == model_id
    
    @patch('research_viz_agent.mcp_tools.huggingface_tool.requests.get')
    def test_medical_cv_search_constructs_correct_query(self, mock_get):
        """Test that medical CV search builds the correct search query."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        tool = HuggingFaceTool()
        tool.search_medical_cv_models(additional_query="pathology")
        
        call_args = mock_get.call_args
        search_param = call_args[1]['params']['search']
        
        # Should include base medical imaging query plus additional terms
        assert 'medical imaging' in search_param
        assert 'pathology' in search_param
