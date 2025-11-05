"""
Unit tests for LLM factory.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from research_viz_agent.utils.llm_factory import LLMFactory


class TestLLMFactoryCreateLLM:
    """Tests for LLMFactory.create_llm method."""
    
    def test_create_llm_none_provider(self):
        """Test creating LLM with 'none' provider returns None."""
        llm = LLMFactory.create_llm(provider="none")
        assert llm is None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_create_openai_llm_with_env_var(self):
        """Test creating OpenAI LLM with environment variable."""
        llm = LLMFactory.create_llm(provider="openai")
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-3.5-turbo"
        assert llm.temperature == 0.7
    
    def test_create_openai_llm_with_api_key(self):
        """Test creating OpenAI LLM with explicit API key."""
        llm = LLMFactory.create_llm(
            provider="openai",
            api_key="test-key-123",
            model_name="gpt-4o",
            temperature=0.5
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.5
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_openai_llm_without_api_key_raises_error(self):
        """Test creating OpenAI LLM without API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key must be provided"):
            LLMFactory.create_llm(provider="openai")
    
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-github-token"})
    def test_create_github_llm_with_env_var(self):
        """Test creating GitHub LLM with environment variable."""
        llm = LLMFactory.create_llm(provider="github")
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.7
    
    def test_create_github_llm_with_api_key(self):
        """Test creating GitHub LLM with explicit API key."""
        llm = LLMFactory.create_llm(
            provider="github",
            api_key="ghp_test123",
            model_name="Llama-3.2-11B-Vision-Instruct",
            temperature=0.3
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "Llama-3.2-11B-Vision-Instruct"
        assert llm.temperature == 0.3
    
    def test_create_github_llm_with_custom_base_url(self):
        """Test creating GitHub LLM with custom base URL."""
        llm = LLMFactory.create_llm(
            provider="github",
            api_key="ghp_test123",
            base_url="https://custom.api.endpoint"
        )
        assert isinstance(llm, ChatOpenAI)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_github_llm_without_token_raises_error(self):
        """Test creating GitHub LLM without token raises ValueError."""
        with pytest.raises(ValueError, match="GitHub token must be provided"):
            LLMFactory.create_llm(provider="github")
    
    def test_create_llm_invalid_provider_raises_error(self):
        """Test creating LLM with invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            LLMFactory.create_llm(provider="invalid")


class TestLLMFactoryGetAvailableModels:
    """Tests for LLMFactory.get_available_models method."""
    
    def test_get_openai_models(self):
        """Test getting available OpenAI models."""
        models = LLMFactory.get_available_models("openai")
        assert isinstance(models, dict)
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models
        assert "description" in models["gpt-4o"]
        assert "context_window" in models["gpt-4o"]
        assert "cost_per_1k_tokens" in models["gpt-4o"]
    
    def test_get_github_models(self):
        """Test getting available GitHub models."""
        models = LLMFactory.get_available_models("github")
        assert isinstance(models, dict)
        assert "gpt-4o" in models
        assert "Llama-3.2-11B-Vision-Instruct" in models
        assert "Phi-3.5-mini-instruct" in models
        assert "Mistral-large-2407" in models
        
        # Check structure
        assert "description" in models["gpt-4o"]
        assert "provider" in models["gpt-4o"]
        assert "cost" in models["gpt-4o"]
        assert models["gpt-4o"]["cost"] == "Free for GitHub Pro users"
    
    def test_get_models_for_none_provider(self):
        """Test getting models for 'none' provider returns empty dict."""
        models = LLMFactory.get_available_models("none")
        assert models == {}
    
    def test_get_models_for_invalid_provider(self):
        """Test getting models for invalid provider returns empty dict."""
        models = LLMFactory.get_available_models("invalid")
        assert models == {}


class TestLLMFactoryValidateProviderConfig:
    """Tests for LLMFactory.validate_provider_config method."""
    
    def test_validate_none_provider(self):
        """Test validating 'none' provider always succeeds."""
        is_valid, message = LLMFactory.validate_provider_config("none")
        assert is_valid is True
        assert "No LLM provider selected" in message
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_validate_openai_with_key(self):
        """Test validating OpenAI with API key succeeds."""
        is_valid, message = LLMFactory.validate_provider_config("openai")
        assert is_valid is True
        assert "OpenAI configuration valid" in message
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_openai_without_key(self):
        """Test validating OpenAI without API key fails."""
        is_valid, message = LLMFactory.validate_provider_config("openai")
        assert is_valid is False
        assert "OpenAI API key not found" in message
        assert "OPENAI_API_KEY" in message
    
    @patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"})
    def test_validate_github_with_token(self):
        """Test validating GitHub with token succeeds."""
        is_valid, message = LLMFactory.validate_provider_config("github")
        assert is_valid is True
        assert "GitHub Models configuration valid" in message
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_github_without_token(self):
        """Test validating GitHub without token fails."""
        is_valid, message = LLMFactory.validate_provider_config("github")
        assert is_valid is False
        assert "GitHub token not found" in message
        assert "GITHUB_TOKEN" in message
        assert "GitHub Pro" in message
    
    def test_validate_invalid_provider(self):
        """Test validating invalid provider fails."""
        is_valid, message = LLMFactory.validate_provider_config("invalid")
        assert is_valid is False
        assert "Unknown provider: invalid" in message


class TestLLMFactoryGetProviderInfo:
    """Tests for LLMFactory.get_provider_info method."""
    
    def test_get_openai_info(self):
        """Test getting OpenAI provider info."""
        info = LLMFactory.get_provider_info("openai")
        assert info["name"] == "OpenAI"
        assert info["cost"] == "Pay-per-use"
        assert info["env_var"] == "OPENAI_API_KEY"
        assert "https://platform.openai.com" in info["setup_url"]
        assert "gpt-4o" in info["models"]
        assert "gpt-3.5-turbo" in info["models"]
    
    def test_get_github_info(self):
        """Test getting GitHub provider info."""
        info = LLMFactory.get_provider_info("github")
        assert info["name"] == "GitHub Models"
        assert "Free with GitHub Pro" in info["cost"]
        assert info["env_var"] == "GITHUB_TOKEN"
        assert "https://github.com/settings/tokens" in info["setup_url"]
        assert "gpt-4o" in info["models"]
        assert "Llama-3.2-11B-Vision-Instruct" in info["models"]
        assert "GitHub Pro" in info["requirements"]
    
    def test_get_none_provider_info(self):
        """Test getting 'none' provider info."""
        info = LLMFactory.get_provider_info("none")
        assert info["name"] == "No LLM"
        assert info["cost"] == "Free"
        assert info["env_var"] is None
        assert info["setup_url"] is None
        assert info["models"] == []
    
    def test_get_invalid_provider_info(self):
        """Test getting invalid provider info returns error."""
        info = LLMFactory.get_provider_info("invalid")
        assert "error" in info
        assert "Unknown provider: invalid" in info["error"]


class TestLLMFactoryCreateEmbeddings:
    """Tests for LLMFactory.create_embeddings method."""
    
    def test_create_embeddings_none_provider(self):
        """Test creating embeddings with 'none' provider returns None."""
        embeddings = LLMFactory.create_embeddings(provider="none")
        assert embeddings is None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_create_openai_embeddings_with_env_var(self):
        """Test creating OpenAI embeddings with environment variable."""
        embeddings = LLMFactory.create_embeddings(provider="openai")
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    def test_create_openai_embeddings_with_api_key(self):
        """Test creating OpenAI embeddings with explicit API key."""
        embeddings = LLMFactory.create_embeddings(
            provider="openai",
            api_key="test-key-456"
        )
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_openai_embeddings_without_key_raises_error(self):
        """Test creating OpenAI embeddings without API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key must be provided"):
            LLMFactory.create_embeddings(provider="openai")
    
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-github-token"})
    def test_create_github_embeddings_with_env_var(self):
        """Test creating GitHub embeddings with environment variable."""
        embeddings = LLMFactory.create_embeddings(provider="github")
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    def test_create_github_embeddings_with_api_key(self):
        """Test creating GitHub embeddings with explicit API key."""
        embeddings = LLMFactory.create_embeddings(
            provider="github",
            api_key="ghp_test789"
        )
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    def test_create_github_embeddings_with_custom_base_url(self):
        """Test creating GitHub embeddings with custom base URL."""
        embeddings = LLMFactory.create_embeddings(
            provider="github",
            api_key="ghp_test789",
            base_url="https://custom.embeddings.endpoint"
        )
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_github_embeddings_without_token_raises_error(self):
        """Test creating GitHub embeddings without token raises ValueError."""
        with pytest.raises(ValueError, match="GitHub token must be provided"):
            LLMFactory.create_embeddings(provider="github")
    
    def test_create_embeddings_invalid_provider_raises_error(self):
        """Test creating embeddings with invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported embeddings provider: invalid"):
            LLMFactory.create_embeddings(provider="invalid")


class TestLLMFactoryPrivateMethods:
    """Tests for private helper methods in LLMFactory."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_llm_defaults(self):
        """Test _create_openai_llm with default parameters."""
        llm = LLMFactory._create_openai_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-3.5-turbo"
        assert llm.temperature == 0.7
    
    def test_create_openai_llm_custom_params(self):
        """Test _create_openai_llm with custom parameters."""
        llm = LLMFactory._create_openai_llm(
            model_name="gpt-4",
            temperature=0.2,
            api_key="custom-key"
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4"
        assert llm.temperature == 0.2
    
    @patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test"})
    def test_create_github_llm_defaults(self):
        """Test _create_github_llm with default parameters."""
        llm = LLMFactory._create_github_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.7
    
    def test_create_github_llm_custom_params(self):
        """Test _create_github_llm with custom parameters."""
        llm = LLMFactory._create_github_llm(
            model_name="Phi-3.5-mini-instruct",
            temperature=0.1,
            api_key="ghp_custom",
            base_url="https://test.endpoint"
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "Phi-3.5-mini-instruct"
        assert llm.temperature == 0.1


class TestLLMFactoryIntegration:
    """Integration tests for LLMFactory."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GITHUB_TOKEN": "ghp_test"})
    def test_all_providers_can_be_created(self):
        """Test that all supported providers can create LLM instances."""
        # None provider
        llm_none = LLMFactory.create_llm(provider="none")
        assert llm_none is None
        
        # OpenAI provider
        llm_openai = LLMFactory.create_llm(provider="openai")
        assert isinstance(llm_openai, ChatOpenAI)
        
        # GitHub provider
        llm_github = LLMFactory.create_llm(provider="github")
        assert isinstance(llm_github, ChatOpenAI)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GITHUB_TOKEN": "ghp_test"})
    def test_all_providers_have_info(self):
        """Test that all providers return valid info."""
        for provider in ["openai", "github", "none"]:
            info = LLMFactory.get_provider_info(provider)
            assert "name" in info
            assert "cost" in info
            assert "models" in info
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "GITHUB_TOKEN": "ghp_test"})
    def test_all_providers_can_validate(self):
        """Test that all providers can be validated."""
        for provider in ["openai", "github", "none"]:
            is_valid, message = LLMFactory.validate_provider_config(provider)
            assert is_valid is True
            assert isinstance(message, str)
    
    def test_model_catalog_completeness(self):
        """Test that model catalogs contain expected structure."""
        for provider in ["openai", "github"]:
            models = LLMFactory.get_available_models(provider)
            assert len(models) > 0
            
            for model_name, model_info in models.items():
                assert isinstance(model_name, str)
                assert isinstance(model_info, dict)
                assert "description" in model_info
                
                if provider == "github":
                    assert "provider" in model_info
                    assert "cost" in model_info
