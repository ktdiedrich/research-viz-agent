#!/usr/bin/env python3
"""
Script to validate LLM provider configurations and test connectivity.
"""
import sys
from dotenv import load_dotenv
from research_viz_agent.utils.llm_factory import LLMFactory, LLMProvider

# Load environment variables
load_dotenv()

def test_llm_provider(provider: LLMProvider, model_name: str = None):
    """Test an LLM provider configuration."""
    print(f"\n{'='*50}")
    print(f"Testing {provider.upper()} Provider")
    print(f"{'='*50}")
    
    # Get provider info
    info = LLMFactory.get_provider_info(provider)
    print(f"Description: {info['description']}")
    if info.get('cost'):
        print(f"Cost: {info['cost']}")
    if info.get('requirements'):
        print(f"Requirements: {info['requirements']}")
    
    # Check configuration
    is_valid, message = LLMFactory.validate_provider_config(provider)
    print(f"Configuration: {'✓' if is_valid else '✗'} {message}")
    
    if not is_valid:
        if info.get('setup_url'):
            print(f"Setup URL: {info['setup_url']}")
        if info.get('env_var'):
            print(f"Set environment variable: {info['env_var']}=your_key_here")
        return False
    
    # Try to create LLM instance
    try:
        print(f"Creating LLM instance...")
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=0.7
        )
        print(f"✓ LLM instance created successfully")
        
        # Test a simple inference
        print(f"Testing inference...")
        from langchain_core.messages import HumanMessage
        
        test_message = [HumanMessage(content="Say 'Hello from medical CV research agent!' in exactly those words.")]
        response = llm.invoke(test_message)
        
        print(f"✓ Inference successful")
        print(f"Response: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False

def main():
    """Test all available LLM providers."""
    print("LLM Provider Configuration Validator")
    print("="*60)
    
    providers_to_test = ["openai", "github"]
    results = {}
    
    for provider in providers_to_test:
        try:
            success = test_llm_provider(provider)
            results[provider] = success
        except Exception as e:
            print(f"✗ {provider.upper()} test failed with exception: {e}")
            results[provider] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working_providers = [p for p, success in results.items() if success]
    
    if working_providers:
        print(f"✓ Working providers: {', '.join(working_providers).upper()}")
        print(f"\nYou can use any of these providers:")
        for provider in working_providers:
            models = list(LLMFactory.get_available_models(provider).keys())[:3]
            print(f"  --llm-provider {provider} --model {models[0]}")
    else:
        print(f"✗ No working providers found")
        print(f"\nTo get started:")
        print(f"1. For OpenAI: Get API key at https://platform.openai.com/api-keys")
        print(f"   export OPENAI_API_KEY=your_openai_key")
        print(f"2. For GitHub Models: Get token at https://github.com/settings/tokens")
        print(f"   export GITHUB_TOKEN=your_github_token")
        print(f"   (Requires GitHub Pro subscription for free access)")
        print(f"3. Or use without AI: --llm-provider none")
    
    # Example commands
    print(f"\nExample commands:")
    if "openai" in working_providers:
        print(f"  research-viz-agent 'lung cancer detection' --llm-provider openai")
    if "github" in working_providers:
        print(f"  research-viz-agent 'skin lesion classification' --llm-provider github --model gpt-4o")
    print(f"  research-viz-agent 'brain tumor MRI' --llm-provider none  # No AI summary")
    
    print()

if __name__ == "__main__":
    main()