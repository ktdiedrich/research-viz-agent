"""
MCP tool for searching and retrieving AI models from HuggingFace.
"""
import requests
from typing import List, Dict, Optional
import os


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
    
    def search_medical_cv_models(self, additional_query: str = "") -> List[Dict]:
        """
        Search for medical computer vision models on HuggingFace.
        
        Args:
            additional_query: Additional search terms to refine the query
            
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
            limit=20
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


def create_huggingface_tool(token: Optional[str] = None) -> HuggingFaceTool:
    """Create and return a HuggingFaceTool instance."""
    return HuggingFaceTool(token=token)
