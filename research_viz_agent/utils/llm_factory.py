"""
Factory for creating LLM instances with support for multiple providers.
"""
import os
from typing import Optional, Union, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LLMProvider = Literal["openai", "github", "none"]


class LLMFactory:
    """Factory for creating LLM instances from different providers."""
    
    @staticmethod
    def create_llm(
        provider: LLMProvider = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[ChatOpenAI]:
        """
        Create an LLM instance based on the specified provider.
        
        Args:
            provider: LLM provider ("openai", "github", or "none")
            model_name: Model name to use
            temperature: Temperature for sampling
            api_key: API key for the provider
            base_url: Base URL for API endpoints
            
        Returns:
            ChatOpenAI instance or None if provider is "none"
        """
        if provider == "none":
            return None
        
        if provider == "openai":
            return LLMFactory._create_openai_llm(model_name, temperature, api_key)
        
        elif provider == "github":
            return LLMFactory._create_github_llm(model_name, temperature, api_key, base_url)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _create_openai_llm(
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        model_name = model_name or "gpt-3.5-turbo"
        
        return ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
    
    @staticmethod
    def _create_github_llm(
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> ChatOpenAI:
        """Create GitHub Models LLM instance using OpenAI-compatible interface."""
        # GitHub Models uses GitHub token for authentication
        api_key = api_key or os.getenv("GITHUB_TOKEN")
        if not api_key:
            raise ValueError(
                "GitHub token must be provided or set in GITHUB_TOKEN environment variable. "
                "Get your token at: https://github.com/settings/tokens"
            )
        
        # GitHub Models API endpoint
        base_url = base_url or "https://models.inference.ai.azure.com"
        
        # Default GitHub model (can be overridden)
        model_name = model_name or "gpt-4o"
        
        # Available GitHub Models:
        # - gpt-4o, gpt-4o-mini (OpenAI)
        # - Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision-Instruct (Meta)
        # - Phi-3.5-mini-instruct, Phi-3.5-MoE-instruct (Microsoft)
        # - Mistral-large-2407, Mistral-Nemo (Mistral AI)
        # - AI21-Jamba-1.5-Large, AI21-Jamba-1.5-Mini (AI21)
        
        return ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
        )
    
    @staticmethod
    def get_available_models(provider: LLMProvider) -> dict:
        """
        Get available models for a provider.
        
        Returns:
            Dictionary with model information
        """
        if provider == "openai":
            return {
                "gpt-4o": {
                    "description": "Most capable GPT-4 model with vision",
                    "context_window": 128000,
                    "cost_per_1k_tokens": {"input": 0.005, "output": 0.015}
                },
                "gpt-4o-mini": {
                    "description": "Faster, cheaper GPT-4 model",
                    "context_window": 128000,
                    "cost_per_1k_tokens": {"input": 0.00015, "output": 0.0006}
                },
                "gpt-3.5-turbo": {
                    "description": "Fast, inexpensive model for simple tasks",
                    "context_window": 16385,
                    "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015}
                }
            }
        
        elif provider == "github":
            return {
                "gpt-4o": {
                    "description": "OpenAI GPT-4o via GitHub Models (FREE for GitHub Pro users)",
                    "context_window": 128000,
                    "provider": "OpenAI",
                    "cost": "Free for GitHub Pro users"
                },
                "gpt-4o-mini": {
                    "description": "OpenAI GPT-4o mini via GitHub Models (FREE for GitHub Pro users)",
                    "context_window": 128000,
                    "provider": "OpenAI",
                    "cost": "Free for GitHub Pro users"
                },
                "Llama-3.2-11B-Vision-Instruct": {
                    "description": "Meta Llama 3.2 11B with vision capabilities (FREE for GitHub Pro users)",
                    "context_window": 8192,
                    "provider": "Meta",
                    "cost": "Free for GitHub Pro users"
                },
                "Llama-3.2-90B-Vision-Instruct": {
                    "description": "Meta Llama 3.2 90B with vision capabilities (FREE for GitHub Pro users)",
                    "context_window": 8192,
                    "provider": "Meta", 
                    "cost": "Free for GitHub Pro users"
                },
                "Phi-3.5-mini-instruct": {
                    "description": "Microsoft Phi-3.5 mini instruct model (FREE for GitHub Pro users)",
                    "context_window": 4096,
                    "provider": "Microsoft",
                    "cost": "Free for GitHub Pro users"
                },
                "Phi-3.5-MoE-instruct": {
                    "description": "Microsoft Phi-3.5 MoE instruct model (FREE for GitHub Pro users)",
                    "context_window": 4096,
                    "provider": "Microsoft",
                    "cost": "Free for GitHub Pro users"
                },
                "Mistral-large-2407": {
                    "description": "Mistral AI large model (FREE for GitHub Pro users)",
                    "context_window": 32768,
                    "provider": "Mistral AI",
                    "cost": "Free for GitHub Pro users"
                },
                "Mistral-Nemo": {
                    "description": "Mistral AI Nemo model (FREE for GitHub Pro users)",
                    "context_window": 32768,
                    "provider": "Mistral AI",
                    "cost": "Free for GitHub Pro users"
                }
            }
        
        else:
            return {}
    
    @staticmethod
    def validate_provider_config(provider: LLMProvider) -> tuple[bool, str]:
        """
        Validate that the provider configuration is correct.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if provider == "none":
            return True, "No LLM provider selected"
        
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False, (
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable or "
                    "get a key from: https://platform.openai.com/api-keys"
                )
            return True, "OpenAI configuration valid"
        
        elif provider == "github":
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                return False, (
                    "GitHub token not found. Set GITHUB_TOKEN environment variable or "
                    "get a token from: https://github.com/settings/tokens\n"
                    "Note: GitHub Models requires GitHub Pro subscription for free access"
                )
            return True, "GitHub Models configuration valid"
        
        else:
            return False, f"Unknown provider: {provider}"
    
    @staticmethod
    def get_provider_info(provider: LLMProvider) -> dict:
        """Get information about a provider."""
        if provider == "openai":
            return {
                "name": "OpenAI",
                "description": "Official OpenAI API with pay-per-use pricing",
                "setup_url": "https://platform.openai.com/api-keys",
                "env_var": "OPENAI_API_KEY",
                "cost": "Pay-per-use",
                "models": list(LLMFactory.get_available_models("openai").keys())
            }
        
        elif provider == "github":
            return {
                "name": "GitHub Models",
                "description": "Free AI models via GitHub (requires GitHub Pro subscription)",
                "setup_url": "https://github.com/settings/tokens",
                "env_var": "GITHUB_TOKEN",
                "cost": "Free with GitHub Pro subscription",
                "models": list(LLMFactory.get_available_models("github").keys()),
                "requirements": "GitHub Pro subscription required for free access"
            }
        
        elif provider == "none":
            return {
                "name": "No LLM",
                "description": "Skip AI summarization, collect results only",
                "setup_url": None,
                "env_var": None,
                "cost": "Free",
                "models": []
            }
        
        else:
            return {"error": f"Unknown provider: {provider}"}
    
    @staticmethod
    def create_embeddings(
        provider: LLMProvider = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[OpenAIEmbeddings]:
        """
        Create an embeddings instance based on the specified provider.
        
        Args:
            provider: Embeddings provider ("openai" or "github")
            api_key: API key for the provider
            base_url: Base URL for API endpoints
            
        Returns:
            OpenAIEmbeddings instance or None if provider is "none"
        """
        if provider == "none":
            return None
        
        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
            return OpenAIEmbeddings(
                api_key=api_key
            )
        
        elif provider == "github":
            # GitHub Models uses GitHub token for authentication
            api_key = api_key or os.getenv("GITHUB_TOKEN")
            if not api_key:
                raise ValueError(
                    "GitHub token must be provided or set in GITHUB_TOKEN environment variable for embeddings. "
                    "Get your token at: https://github.com/settings/tokens"
                )
            
            # GitHub Models API endpoint for embeddings
            base_url = base_url or "https://models.inference.ai.azure.com"
            
            # Use text-embedding-3-small as default embedding model for GitHub
            return OpenAIEmbeddings(
                api_key=api_key,
                base_url=base_url,
                model="text-embedding-3-small"
            )
        
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")