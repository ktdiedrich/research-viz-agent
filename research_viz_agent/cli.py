#!/usr/bin/env python3
"""
Command-line interface for the Medical CV Research Agent.
"""
import argparse
import sys
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description="Research Agent for Medical Computer Vision AI Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regular research
  %(prog)s "lung cancer detection"
  %(prog)s "skin lesion classification" --email your@email.com
  %(prog)s "brain tumor segmentation" --output results.txt
  %(prog)s "lung cancer detection" --display-results 10
  %(prog)s "radiology AI" --max-results 50 --display-results 15
  
  # RAG functionality
  %(prog)s --rag-search "deep learning medical imaging"
  %(prog)s --rag-search "CNN chest x-ray" --rag-source arxiv
  %(prog)s --rag-stats
  %(prog)s "lung cancer" --no-rag
  
  # Cost-saving options
  %(prog)s "lung cancer detection" --no-summary
  %(prog)s "skin lesion" --no-summary --no-rag
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Research query for medical CV AI models"
    )
    
    parser.add_argument(
        "--email",
        type=str,
        default="research@example.com",
        help="Email for PubMed API (default: research@example.com)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to file"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of results to fetch from each source (default: 20)"
    )
    
    parser.add_argument(
        "--display-results",
        type=int,
        default=5,
        help="Number of results to display from each source (default: 5)"
    )
    
    parser.add_argument(
        "--rag-search",
        type=str,
        help="Search the RAG database instead of fetching new results"
    )
    
    parser.add_argument(
        "--rag-source",
        type=str,
        choices=["arxiv", "pubmed", "huggingface"],
        help="Filter RAG search by source"
    )
    
    parser.add_argument(
        "--rag-results",
        type=int,
        default=10,
        help="Number of RAG search results to return (default: 10)"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG storage and search functionality"
    )
    
    parser.add_argument(
        "--rag-stats",
        action="store_true",
        help="Show RAG database statistics and exit"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip AI summarization to avoid OpenAI API costs (collect results only)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        print("Initializing Medical CV Research Agent...")
        
        # Skip OpenAI setup if only collecting results
        if args.no_summary:
            print("⚠ Running without AI summarization (--no-summary flag)")
            # Use a dummy model name, won't be used
            agent = MedicalCVResearchAgent(
                pubmed_email=args.email,
                model_name="gpt-3.5-turbo",  # Won't be used
                temperature=args.temperature,
                max_results=args.max_results,
                enable_rag=not args.no_rag,
                skip_openai_init=True
            )
        else:
            agent = MedicalCVResearchAgent(
                pubmed_email=args.email,
                model_name=args.model,
                temperature=args.temperature,
                max_results=args.max_results,
                enable_rag=not args.no_rag
            )
        
        # Handle RAG statistics request
        if args.rag_stats:
            stats = agent.get_rag_stats()
            if 'error' in stats:
                print(f"Error: {stats['error']}")
            else:
                print(f"\nRAG Database Statistics:")
                print(f"Collection: {stats['collection_name']}")
                print(f"Documents: {stats['document_count']}")
                print(f"Directory: {stats['persist_directory']}")
            return
        
        # Handle RAG search
        if args.rag_search:
            print(f"Searching RAG database for: {args.rag_search}")
            rag_results = agent.search_rag(
                query=args.rag_search,
                k=args.rag_results,
                source_filter=args.rag_source
            )
            formatted_output = agent.format_rag_results(rag_results, show_content=True)
        else:
            # Run regular research
            results = agent.research(args.query)
            formatted_output = agent.format_results(results, display_limit=args.display_results)
        
        # Display results
        print(formatted_output)
        
        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            print(f"\n\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
