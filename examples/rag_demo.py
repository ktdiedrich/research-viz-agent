#!/usr/bin/env python3
"""
Example demonstrating RAG functionality with the Medical CV Research Agent.
"""
import os
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent


def main():
    """Demonstrate RAG functionality."""
    print("="*60)
    print("Medical CV Research Agent - RAG Demo")
    print("="*60)
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ Please set your OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    try:
        # Initialize agent with RAG enabled
        print("Initializing agent with RAG functionality...")
        agent = MedicalCVResearchAgent(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_results=15,
            enable_rag=True
        )
        
        # Perform initial research to populate RAG database
        queries = [
            "lung cancer detection",
            "skin lesion classification",
            "brain tumor segmentation"
        ]
        
        print("\n" + "="*60)
        print("Phase 1: Populating RAG Database")
        print("="*60)
        
        for query in queries:
            print(f"\nResearching: {query}")
            results = agent.research(query)
            print(f"Found {results['total_papers']} papers and {results['total_models']} models")
        
        # Show RAG statistics
        print("\n" + "="*60)
        print("Phase 2: RAG Database Statistics")
        print("="*60)
        
        stats = agent.get_rag_stats()
        if 'error' not in stats:
            print(f"Total documents in RAG: {stats['document_count']}")
            print(f"Collection: {stats['collection_name']}")
            print(f"Directory: {stats['persist_directory']}")
        else:
            print(f"Error getting stats: {stats['error']}")
        
        # Demonstrate RAG search
        print("\n" + "="*60)
        print("Phase 3: RAG Search Demonstration")
        print("="*60)
        
        # Search across all sources
        print("\n--- Search: 'deep learning medical imaging' ---")
        rag_results = agent.search_rag("deep learning medical imaging", k=5)
        print(agent.format_rag_results(rag_results))
        
        # Search specific source
        print("\n--- Search ArXiv only: 'CNN classification' ---")
        rag_results = agent.search_rag("CNN classification", k=3, source_filter="arxiv")
        print(agent.format_rag_results(rag_results))
        
        # Search HuggingFace models
        print("\n--- Search HuggingFace models: 'medical' ---")
        rag_results = agent.search_rag("medical", k=3, source_filter="huggingface")
        print(agent.format_rag_results(rag_results, show_content=True))
        
        print("\n" + "="*60)
        print("RAG Demo Complete!")
        print("="*60)
        print()
        print("The RAG database is now populated and can be used for:")
        print("- Fast similarity search across all research sources")
        print("- Historical query analysis")
        print("- Cross-reference research findings")
        print("- Offline search without API calls")
        print()
        print("Use the CLI for more RAG operations:")
        print("  python -m research_viz_agent.cli --rag-search 'your query'")
        print("  python -m research_viz_agent.cli --rag-stats")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())