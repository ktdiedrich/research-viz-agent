"""
Standalone script to test individual MCP tools without requiring OpenAI API.
"""
import sys
from typing import Final


def test_arxiv():
    """Test arXiv tool."""
    print("\n" + "="*60)
    print("Testing arXiv Tool")
    print("="*60 + "\n")
    
    from research_viz_agent.mcp_tools.arxiv_tool import ArxivTool
    
    tool = ArxivTool()
    print("✓ ArxivTool initialized")
    MAX_RESULTS: Final[int] = 6
    print("\nSearching for 'medical imaging deep learning'...")
    results = tool.search_papers("medical imaging deep learning", max_results=MAX_RESULTS)
    
    print(f"\nFound {len(results)} papers:")
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:MAX_RESULTS])}")
        print(f"   URL: {paper['pdf_url']}")
        print(f"   Categories: {', '.join(paper['categories'])}")
    
    return len(results) > 0


def test_pubmed():
    """Test PubMed tool."""
    print("\n" + "="*60)
    print("Testing PubMed Tool")
    print("="*60 + "\n")
    
    from research_viz_agent.mcp_tools.pubmed_tool import PubMedTool
    
    tool = PubMedTool(email="test@example.com")
    print("✓ PubMedTool initialized")
    
    print("\nSearching for 'radiology deep learning'...")
    results = tool.search_papers("radiology deep learning", max_results=3)
    
    print(f"\nFound {len(results)} papers:")
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   PMID: {paper['pmid']}")
        print(f"   Journal: {paper['journal']}")
        abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
        print(f"   Abstract: {abstract}")
    
    return len(results) > 0


def test_huggingface():
    """Test HuggingFace tool."""
    print("\n" + "="*60)
    print("Testing HuggingFace Tool")
    print("="*60 + "\n")
    
    from research_viz_agent.mcp_tools.huggingface_tool import HuggingFaceTool
    
    tool = HuggingFaceTool()
    print("✓ HuggingFaceTool initialized")
    
    print("\nSearching for 'medical imaging' models...")
    results = tool.search_models("medical imaging", limit=5)
    
    print(f"\nFound {len(results)} models:")
    for i, model in enumerate(results, 1):
        print(f"\n{i}. {model['model_id']}")
        print(f"   Author: {model['author']}")
        print(f"   Task: {model['pipeline_tag']}")
        print(f"   Downloads: {model['downloads']}")
        print(f"   URL: {model['model_card_url']}")
    
    return len(results) > 0


def main():
    """Run all tool tests."""
    print("\n" + "="*60)
    print("MCP Tools Test Suite")
    print("="*60)
    print("\nThis script tests each MCP tool independently.")
    print("No OpenAI API key required.\n")
    
    results = {}
    
    try:
        results['arxiv'] = test_arxiv()
    except Exception as e:
        print(f"\n✗ arXiv test failed: {e}")
        results['arxiv'] = False
    
    try:
        results['pubmed'] = test_pubmed()
    except Exception as e:
        print(f"\n✗ PubMed test failed: {e}")
        results['pubmed'] = False
    
    try:
        results['huggingface'] = test_huggingface()
    except Exception as e:
        print(f"\n✗ HuggingFace test failed: {e}")
        results['huggingface'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60 + "\n")
    
    for tool, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{tool:15s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
