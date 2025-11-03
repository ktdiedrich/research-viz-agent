#!/usr/bin/env python3
"""
Script to manage ChromaDB instances for different LLM providers.
"""
import os
import shutil
import argparse
import time
from research_viz_agent.utils.llm_factory import LLMFactory

def backup_chromadb(source_dir: str, backup_dir: str):
    """Backup ChromaDB directory."""
    if os.path.exists(source_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(source_dir, backup_dir)
        print(f"✓ Backed up {source_dir} to {backup_dir}")
    else:
        print(f"⚠ Source directory {source_dir} does not exist")

def restore_chromadb(backup_dir: str, target_dir: str):
    """Restore ChromaDB directory from backup."""
    if os.path.exists(backup_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(backup_dir, target_dir)
        print(f"✓ Restored {backup_dir} to {target_dir}")
    else:
        print(f"⚠ Backup directory {backup_dir} does not exist")

def clear_chromadb(directory: str):
    """Clear ChromaDB directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"✓ Cleared ChromaDB directory: {directory}")
    else:
        print(f"⚠ Directory {directory} does not exist")

def list_chromadb_instances():
    """List all ChromaDB instances."""
    base_patterns = ["./chroma_db*", "./rag_*", "./*chroma*"]
    found_dirs = []
    
    for pattern in base_patterns:
        from glob import glob
        found_dirs.extend(glob(pattern))
    
    # Also check for provider-specific directories
    providers = ["openai", "github"]
    for provider in providers:
        provider_dir = f"./chroma_db_{provider}"
        if os.path.exists(provider_dir):
            found_dirs.append(provider_dir)
    
    if found_dirs:
        print("Found ChromaDB instances:")
        for directory in found_dirs:
            size = get_directory_size(directory)
            print(f"  {directory} ({size})")
    else:
        print("No ChromaDB instances found")

def get_directory_size(directory: str) -> str:
    """Get human-readable directory size."""
    try:
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(directory)
            for filename in filenames
        )
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"
    except:
        return "unknown size"

def setup_provider_specific_rag():
    """Set up provider-specific RAG directories."""
    providers = ["openai", "github"]
    
    # Backup existing if it exists
    if os.path.exists("./chroma_db"):
        backup_chromadb("./chroma_db", "./chroma_db_backup")
        print("ℹ Original ChromaDB backed up to ./chroma_db_backup")
    
    # Create provider-specific directories
    for provider in providers:
        provider_dir = f"./chroma_db_{provider}"
        os.makedirs(provider_dir, exist_ok=True)
        print(f"✓ Created RAG directory for {provider}: {provider_dir}")
    
    print("\nNow you can use provider-specific RAG:")
    print("  --llm-provider openai  → uses ./chroma_db_openai")
    print("  --llm-provider github  → uses ./chroma_db_github")

def main():
    parser = argparse.ArgumentParser(description="Manage ChromaDB instances for different LLM providers")
    parser.add_argument("--list", action="store_true", help="List all ChromaDB instances")
    parser.add_argument("--clear", type=str, help="Clear specified ChromaDB directory")
    parser.add_argument("--clear-all", action="store_true", help="Clear all ChromaDB instances")
    parser.add_argument("--backup", type=str, help="Backup ChromaDB directory")
    parser.add_argument("--restore", type=str, help="Restore ChromaDB directory")
    parser.add_argument("--setup-providers", action="store_true", help="Set up provider-specific RAG directories")
    
    args = parser.parse_args()
    
    if args.list:
        list_chromadb_instances()
    
    elif args.clear:
        clear_chromadb(args.clear)
    
    elif args.clear_all:
        from glob import glob
        patterns = ["./chroma_db*", "./rag_*", "./*chroma*"]
        for pattern in patterns:
            for directory in glob(pattern):
                clear_chromadb(directory)
    
    elif args.backup:
        backup_dir = f"{args.backup}_backup_{int(time.time())}"
        backup_chromadb(args.backup, backup_dir)
    
    elif args.restore:
        target_dir = args.restore.replace("_backup", "").split("_")[0]
        restore_chromadb(args.restore, target_dir)
    
    elif args.setup_providers:
        setup_provider_specific_rag()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()