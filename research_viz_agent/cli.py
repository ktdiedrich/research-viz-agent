#!/usr/bin/env python3
"""
Command-line interface for the Medical CV Research Agent.
"""
import argparse
import sys
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent
from research_viz_agent.utils.llm_factory import LLMFactory, LLMProvider
from research_viz_agent.utils.rag_tracker import RAGTracker, create_bar_chart_ascii
from research_viz_agent.utils.csv_export import export_rag_results_to_csv, export_research_results_to_csv


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description="Research Agent for Medical Computer Vision AI Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regular research with different LLM providers
  %(prog)s "lung cancer detection"
  %(prog)s "skin lesion classification" --llm-provider github --model gpt-4o
  %(prog)s "brain tumor segmentation" --llm-provider openai --model gpt-4o-mini
  %(prog)s "lung cancer detection" --display-results 10
  %(prog)s "radiology AI" --max-results 50 --display-results 15
  
  # GitHub Models (free with GitHub Pro)
  %(prog)s "medical imaging AI" --llm-provider github --model Llama-3.2-11B-Vision-Instruct
  %(prog)s "chest x-ray classification" --llm-provider github --model Phi-3.5-mini-instruct
  
  # RAG functionality
  %(prog)s --rag-search "deep learning medical imaging"
  %(prog)s --rag-search "CNN chest x-ray" --rag-source arxiv
  %(prog)s --rag-search "deep learning" --csv rag_results.csv
  %(prog)s --rag-stats
  %(prog)s "lung cancer" --no-rag
  
  # Export results to CSV
  %(prog)s "lung cancer detection" --csv results.csv
  %(prog)s --rag-search "medical imaging" --csv search.csv --rag-results 50
  
  # Cost-saving and model options
  %(prog)s "lung cancer detection" --no-summary
  %(prog)s "skin lesion" --llm-provider none
  %(prog)s --list-models openai
  %(prog)s --list-models github
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Research query for medical CV AI models"
    )
    
    parser.add_argument(
        "--email",
        type=str,
        default="research@example.com",
        help="Email for PubMed API (default: research@example.com)"
    )
    
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "github", "none"],
        default="github",
        help="LLM provider to use (default: github)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (provider-specific, uses defaults if not specified)"
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
        help="Save results to file (text format)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Export results to CSV file"
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
        "--rag-dir",
        type=str,
        help="Custom RAG storage directory (default: provider-specific directories)"
    )
    
    parser.add_argument(
        "--rag-stats",
        action="store_true",
        help="Show RAG database statistics and exit"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip AI summarization to avoid API costs (same as --llm-provider none)"
    )
    
    parser.add_argument(
        "--list-models",
        type=str,
        choices=["openai", "github"],
        help="List available models for a provider and exit"
    )
    
    parser.add_argument(
        "--provider-info",
        type=str,
        choices=["openai", "github"],
        help="Show provider information and setup instructions"
    )
    
    parser.add_argument(
        "--show-tracking",
        action="store_true",
        help="Show RAG query tracking chart and exit"
    )
    
    parser.add_argument(
        "--tracking-summary",
        action="store_true",
        help="Show RAG tracking summary statistics and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Handle informational commands first
        if args.list_models:
            models = LLMFactory.get_available_models(args.list_models)
            print(f"\n{args.list_models.upper()} Available Models:")
            print("=" * 50)
            for model_name, info in models.items():
                print(f"\n• {model_name}")
                print(f"  Description: {info['description']}")
                if 'provider' in info:
                    print(f"  Provider: {info['provider']}")
                if 'context_window' in info:
                    print(f"  Context Window: {info['context_window']:,} tokens")
                if 'cost' in info:
                    print(f"  Cost: {info['cost']}")
                elif 'cost_per_1k_tokens' in info:
                    cost = info['cost_per_1k_tokens']
                    print(f"  Cost: ${cost['input']:.5f} input / ${cost['output']:.5f} output per 1K tokens")
            print()
            return
        
        if args.provider_info:
            info = LLMFactory.get_provider_info(args.provider_info)
            print(f"\n{info['name']} Provider Information:")
            print("=" * 50)
            print(f"Description: {info['description']}")
            print(f"Cost: {info['cost']}")
            if info['setup_url']:
                print(f"Setup: {info['setup_url']}")
            if info['env_var']:
                print(f"Environment Variable: {info['env_var']}")
            if 'requirements' in info:
                print(f"Requirements: {info['requirements']}")
            print(f"Available Models: {', '.join(info['models'])}")
            
            # Check configuration
            is_valid, message = LLMFactory.validate_provider_config(args.provider_info)
            print(f"\nConfiguration Status: {'✓' if is_valid else '✗'} {message}")
            print()
            return
        
        # Handle tracking commands
        if args.show_tracking or args.tracking_summary:
            rag_dir = args.rag_dir or "./chroma_db"
            # Determine provider-specific directory
            llm_provider = args.llm_provider
            if llm_provider == "github":
                rag_dir = f"{rag_dir}_github"
            elif llm_provider == "openai":
                rag_dir = f"{rag_dir}"
            
            tracking_file = f"{rag_dir}/rag_tracking.json"
            tracker = RAGTracker(tracking_file=tracking_file)
            
            if args.tracking_summary:
                summary = tracker.get_summary()
                print("\n" + "=" * 60)
                print("RAG STORE TRACKING SUMMARY")
                print("=" * 60)
                print(f"Total Queries: {summary['total_queries']}")
                print(f"Total Records: {summary['total_records']}")
                print(f"  - ArXiv: {summary['total_arxiv']}")
                print(f"  - PubMed: {summary['total_pubmed']}")
                print(f"  - HuggingFace: {summary['total_huggingface']}")
                print("=" * 60)
                print(f"\nTracking file: {tracking_file}")
            else:
                queries = tracker.get_all_queries()
                print(create_bar_chart_ascii(queries))
                summary = tracker.get_summary()
                print(f"\nTotal: {summary['total_queries']} queries, {summary['total_records']} records")
                print(f"Tracking file: {tracking_file}")
            return
        
        # Require query for research operations
        if not args.query and not args.rag_search and not args.rag_stats:
            parser.error("Research query is required (or use --rag-search, --rag-stats, --list-models, --provider-info)")
        
        # Determine LLM provider
        llm_provider = args.llm_provider
        if args.no_summary:
            llm_provider = "none"
        
        # Validate provider configuration
        is_valid, message = LLMFactory.validate_provider_config(llm_provider)
        if not is_valid and llm_provider != "none":
            print(f"⚠ {message}")
            print("  Falling back to no-summary mode...")
            llm_provider = "none"
        
        # Initialize agent
        print("Initializing Medical CV Research Agent...")
        
        agent = MedicalCVResearchAgent(
            llm_provider=llm_provider,
            pubmed_email=args.email,
            model_name=args.model,
            temperature=args.temperature,
            max_results=args.max_results,
            enable_rag=not args.no_rag,
            rag_persist_dir=args.rag_dir or "./chroma_db"
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
            
            # Export to CSV if requested
            if args.csv:
                try:
                    export_rag_results_to_csv(rag_results, args.csv)
                    print(f"\nResults exported to {args.csv}")
                except Exception as e:
                    print(f"Warning: Failed to export CSV: {e}", file=sys.stderr)
        else:
            # Run regular research
            results = agent.research(args.query)
            formatted_output = agent.format_results(results, display_limit=args.display_results)
            
            # Export to CSV if requested
            if args.csv:
                try:
                    export_research_results_to_csv(results, args.csv)
                    print(f"\nResults exported to {args.csv}")
                except Exception as e:
                    print(f"Warning: Failed to export CSV: {e}", file=sys.stderr)
        
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
