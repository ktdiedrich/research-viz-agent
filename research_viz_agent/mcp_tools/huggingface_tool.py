"""
MCP tool for searching and retrieving AI models from HuggingFace.
"""
import requests
from typing import List, Dict, Optional
import os
import mcp.types as types
from mcp.server import Server


class HuggingFaceTool:
    """Tool for searching HuggingFace models related to medical computer vision."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace tool.
        
        Args:
            token: Optional HuggingFace API token for authenticated requests
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.base_url = "https://huggingface.co/api"
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    def search_models(
        self,
        search_query: str = "",
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search for models on HuggingFace.
        
        Args:
            search_query: Search query string
            task: Filter by task (e.g., 'image-classification', 'object-detection')
            tags: List of tags to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of model dictionaries with metadata
        """
        url = f"{self.base_url}/models"
        
        params = {
            "limit": limit,
            "full": "true"
        }
        
        if search_query:
            params["search"] = search_query
        
        if task:
            params["filter"] = task
        
        if tags:
            params["tags"] = ",".join(tags)
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            models = response.json()
            
            results = []
            for model in models:
                results.append({
                    'model_id': model.get('id', ''),
                    'author': model.get('author', ''),
                    'model_name': model.get('modelId', ''),
                    'downloads': model.get('downloads', 0),
                    'likes': model.get('likes', 0),
                    'tags': model.get('tags', []),
                    'pipeline_tag': model.get('pipeline_tag', ''),
                    'created_at': model.get('createdAt', ''),
                    'last_modified': model.get('lastModified', ''),
                    'library_name': model.get('library_name', ''),
                    'model_card_url': f"https://huggingface.co/{model.get('id', '')}",
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching HuggingFace: {e}")
            return []
    
    def search_medical_cv_models(self, additional_query: str = "", max_results: int = 20) -> List[Dict]:
        """
        Search for medical computer vision models on HuggingFace.
        
        Args:
            additional_query: Additional search terms to refine the query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant models
        """
        # Common tasks for medical computer vision
        medical_tags = [
            "medical",
            "radiology",
            "pathology",
            "xray",
            "ct-scan",
            "mri",
            "medical-imaging"
        ]
        
        # Search with medical-related query
        search_query = "medical imaging"
        if additional_query:
            search_query = f"{search_query} {additional_query}"
        
        return self.search_models(
            search_query=search_query,
            task="image-classification",  # Start with image classification
            limit=max_results
        )
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The model ID (e.g., 'microsoft/resnet-50')
            
        Returns:
            Model information dictionary
        """
        url = f"{self.base_url}/models/{model_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None


# MCP Server instance
server = Server("huggingface-tool")

# Tool instance
huggingface_tool = HuggingFaceTool()

@server.call_tool()
async def search_huggingface_models(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for AI models on HuggingFace with specified criteria.
    
    Args:
        search_query: Search query string (optional)
        task: Filter by task type (e.g., 'image-classification', 'object-detection') (optional)
        tags: Comma-separated list of tags to filter by (optional)
        limit: Maximum number of results (default: 20)
    """
    search_query = arguments.get("search_query", "")
    task = arguments.get("task")
    tags_str = arguments.get("tags", "")
    limit = arguments.get("limit", 20)
    
    # Parse tags if provided
    tags = [tag.strip() for tag in tags_str.split(",")] if tags_str else None
    
    try:
        results = huggingface_tool.search_models(
            search_query=search_query,
            task=task,
            tags=tags,
            limit=limit
        )
        
        if not results:
            return [types.TextContent(
                type="text",
                text="No models found matching the search criteria."
            )]
        
        # Format results as text
        formatted_results = []
        for i, model in enumerate(results, 1):
            model_text = f"""
{i}. {model['model_id']}
   Author: {model['author']}
   Task: {model['pipeline_tag']}
   Downloads: {model['downloads']:,}
   Likes: {model['likes']}
   Tags: {', '.join(model['tags'][:5])}{'...' if len(model['tags']) > 5 else ''}
   Library: {model['library_name']}
   URL: {model['model_card_url']}
   
"""
            formatted_results.append(model_text)
        
        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} HuggingFace models:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching HuggingFace models: {str(e)}"
        )]

@server.call_tool()
async def search_medical_cv_models(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Search for medical computer vision models on HuggingFace.
    
    Args:
        additional_query: Additional search terms to refine the query (optional)
    """
    additional_query = arguments.get("additional_query", "")
    
    try:
        results = huggingface_tool.search_medical_cv_models(additional_query)
        
        if not results:
            return [types.TextContent(
                type="text",
                text="No medical computer vision models found."
            )]
        
        # Format results as text
        formatted_results = []
        for i, model in enumerate(results, 1):
            model_text = f"""
{i}. {model['model_id']}
   Author: {model['author']}
   Task: {model['pipeline_tag']}
   Downloads: {model['downloads']:,}
   Likes: {model['likes']}
   Tags: {', '.join(model['tags'][:5])}{'...' if len(model['tags']) > 5 else ''}
   Library: {model['library_name']}
   URL: {model['model_card_url']}
   
"""
            formatted_results.append(model_text)
        
        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} medical CV models:\n\n" + "\n".join(formatted_results)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error searching medical CV models: {str(e)}"
        )]

@server.call_tool()
async def get_model_details(
    arguments: Dict
) -> List[types.TextContent]:
    """
    Get detailed information about a specific HuggingFace model.
    
    Args:
        model_id: The model ID (e.g., 'microsoft/resnet-50')
    """
    model_id = arguments.get("model_id", "")
    
    if not model_id:
        return [types.TextContent(
            type="text",
            text="Error: model_id parameter is required"
        )]
    
    try:
        model_info = huggingface_tool.get_model_info(model_id)
        
        if not model_info:
            return [types.TextContent(
                type="text",
                text=f"Model '{model_id}' not found or error retrieving information."
            )]
        
        # Format model details
        details = f"""
Model Details for: {model_info.get('id', model_id)}

Author: {model_info.get('author', 'N/A')}
Task: {model_info.get('pipeline_tag', 'N/A')}
Downloads: {model_info.get('downloads', 0):,}
Likes: {model_info.get('likes', 0)}
Created: {model_info.get('createdAt', 'N/A')}
Last Modified: {model_info.get('lastModified', 'N/A')}
Library: {model_info.get('library_name', 'N/A')}

Tags: {', '.join(model_info.get('tags', []))}

Model Card URL: https://huggingface.co/{model_info.get('id', model_id)}
"""
        
        return [types.TextContent(
            type="text",
            text=details
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting model details: {str(e)}"
        )]

def create_huggingface_tool(token: Optional[str] = None) -> HuggingFaceTool:
    """Create and return a HuggingFaceTool instance."""
    return HuggingFaceTool(token=token)

def get_huggingface_server() -> Server:
    """Get the MCP server instance for HuggingFace tools."""
    return server
