#!/usr/bin/env python3
"""
Convenience script to run research with automatic ChromaDB conflict resolution.
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_research_with_auto_rag(query: str, provider: str = "github", model: str = None, **kwargs):
    """Run research with automatic RAG conflict resolution."""
    
    # Generate unique RAG directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_dir = f"./rag_{provider}_{timestamp}"
    
    # Build command
    cmd = [
        "uv", "run", "python", "-m", "research_viz_agent.cli",
        query,
        "--llm-provider", provider,
        "--rag-dir", rag_dir
    ]
    
    if model:
        cmd.extend(["--model", model])
    
    # Add other arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Using RAG directory: {rag_dir}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Research completed successfully!")
        print(f"📁 RAG data stored in: {rag_dir}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Research failed with error: {e}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Run research with automatic RAG conflict resolution")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--provider", choices=["openai", "github", "none"], default="github", help="LLM provider")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--max-results", type=int, default=20, help="Max results to fetch")
    parser.add_argument("--display-results", type=int, default=5, help="Results to display")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    
    args = parser.parse_args()
    
    if args.no_rag:
        # If no RAG, just run normally
        cmd = [
            "uv", "run", "python", "-m", "research_viz_agent.cli",
            args.query,
            "--llm-provider", args.provider,
            "--no-rag"
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.max_results != 20:
            cmd.extend(["--max-results", str(args.max_results)])
        if args.display_results != 5:
            cmd.extend(["--display-results", str(args.display_results)])
        if args.output:
            cmd.extend(["--output", args.output])
        
        subprocess.run(cmd)
    else:
        # Use auto RAG resolution
        kwargs = {
            "max_results": args.max_results if args.max_results != 20 else None,
            "display_results": args.display_results if args.display_results != 5 else None,
            "output": args.output
        }
        
        run_research_with_auto_rag(
            args.query, 
            provider=args.provider,
            model=args.model,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

if __name__ == "__main__":
    main()