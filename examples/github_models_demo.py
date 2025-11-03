#!/usr/bin/env python3
"""
Example script demonstrating GitHub Models integration with the research agent.
"""
import os
from dotenv import load_dotenv
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
from research_viz_agent.utils.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

def main():
    """Demonstrate GitHub Models integration."""
    
    print("Medical CV Research Agent - GitHub Models Demo")
    print("=" * 60)
    
    # Check available providers
    print("\n1. Available LLM Providers:")
    for provider in ["openai", "github"]:
        info = LLMFactory.get_provider_info(provider)
        is_valid, message = LLMFactory.validate_provider_config(provider)
        status = "✓" if is_valid else "✗"
        print(f"   {status} {info['name']}: {message}")
    
    # Show GitHub Models
    print("\n2. GitHub Models (Free with GitHub Pro):")
    github_models = LLMFactory.get_available_models("github")
    for model_name, info in list(github_models.items())[:5]:  # Show first 5
        print(f"   • {model_name} ({info['provider']})")
    print(f"   ... and {len(github_models) - 5} more models")
    
    # Example research with GitHub Models
    print("\n3. Example Research with GitHub Models:")
    
    # Check if GitHub token is available
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("   ⚠ GITHUB_TOKEN not set. Set it to try GitHub Models:")
        print("   export GITHUB_TOKEN=your_github_token_here")
        print("   Get token at: https://github.com/settings/tokens")
        return
    
    try:
        # Initialize agent with GitHub Models
        agent = MedicalCVResearchAgent(
            llm_provider="github",
            model_name="gpt-4o-mini",  # Free GPT-4o mini via GitHub
            max_results=5,  # Limit for demo
            enable_rag=False  # Disable RAG for simple demo
        )
        
        # Run a quick research
        query = "lung nodule detection CT"
        print(f"   Researching: {query}")
        
        results = agent.research(query)
        
        print(f"\n   Results Summary:")
        print(f"   - ArXiv papers: {len(results['arxiv_results'])}")
        print(f"   - PubMed papers: {len(results['pubmed_results'])}")
        print(f"   - HuggingFace models: {len(results['huggingface_results'])}")
        
        # Show a snippet of the AI summary
        summary = results['summary']
        if summary and len(summary) > 200:
            print(f"\n   AI Summary (first 200 chars):")
            print(f"   {summary[:200]}...")
        elif summary:
            print(f"\n   AI Summary:")
            print(f"   {summary}")
        
    except Exception as e:
        print(f"   ⚠ Demo failed: {e}")
    
    print(f"\n4. CLI Examples:")
    print(f"   # Use GitHub Models (free with GitHub Pro)")
    print(f"   research-viz-agent 'brain tumor MRI' --llm-provider github --model gpt-4o")
    print(f"   research-viz-agent 'skin cancer' --llm-provider github --model Llama-3.2-11B-Vision-Instruct")
    print(f"   ")
    print(f"   # Compare providers")
    print(f"   research-viz-agent --list-models github")
    print(f"   research-viz-agent --provider-info github")
    print(f"   ")
    print(f"   # Cost-free research (no AI summary)")
    print(f"   research-viz-agent 'medical imaging' --llm-provider none")

if __name__ == "__main__":
    main()