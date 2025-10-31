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
  %(prog)s "lung cancer detection"
  %(prog)s "skin lesion classification" --email your@email.com
  %(prog)s "brain tumor segmentation" --output results.txt
  %(prog)s "lung cancer detection" --display-results 10
  %(prog)s "radiology AI" --max-results 50 --display-results 15
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
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        print("Initializing Medical CV Research Agent...")
        agent = MedicalCVResearchAgent(
            pubmed_email=args.email,
            model_name=args.model,
            temperature=args.temperature,
            max_results=args.max_results
        )
        
        # Run research
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
