"""
Example: Agent Client

This example shows how to use the agent client to communicate
with a running research agent server.
"""
import asyncio
from research_viz_agent.agent_protocol.client import AgentClient


def sync_example():
    """Example using synchronous client."""
    print("🔄 Synchronous Client Example\n")
    
    # Connect to the agent server
    with AgentClient(base_url="http://localhost:8000") as client:
        # Check agent status
        print("1. Getting agent status...")
        status = client.get_status()
        print(f"   Agent: {status.agent_name} v{status.version}")
        print(f"   Status: {status.status}")
        print(f"   Capabilities: {len(status.capabilities)}")
        
        # Perform research
        print("\n2. Requesting research on 'lung cancer detection'...")
        result = client.research(
            query="lung cancer detection deep learning",
            max_results=10,
            enable_rag=True
        )
        print(f"   Found {result.total_papers} papers, {result.total_models} models")
        print(f"   Summary: {result.summary[:200]}...")
        
        # Search RAG database
        print("\n3. Searching RAG database...")
        rag_results = client.search_rag(
            query="convolutional neural networks",
            k=5
        )
        print(f"   Found {rag_results.total_count} relevant documents")
        
        # Health check
        print("\n4. Health check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")


async def async_example():
    """Example using asynchronous client."""
    print("\n⚡ Asynchronous Client Example\n")
    
    # Connect to the agent server
    async with AgentClient(base_url="http://localhost:8000") as client:
        # Get status and perform research concurrently
        print("1. Executing concurrent requests...")
        status_task = client.get_status_async()
        research_task = client.research_async(
            query="brain tumor segmentation",
            max_results=5
        )
        
        status, result = await asyncio.gather(status_task, research_task)
        
        print(f"   Agent: {status.agent_name}")
        print(f"   Research: {result.total_papers} papers found")
        
        # Generic request using standard protocol
        print("\n2. Using generic request format...")
        response = await client.send_request_async(
            capability="search_rag",
            parameters={
                "query": "medical imaging AI",
                "k": 3
            }
        )
        print(f"   Status: {response.status}")
        print(f"   Found: {response.result['total_count']} results")


def main():
    """Run examples."""
    print("=" * 60)
    print("Agent Client Examples")
    print("=" * 60)
    print("\nMake sure the agent server is running:")
    print("  python examples/agent_server.py")
    print("\nOr start it with:")
    print("  research-viz-agent serve\n")
    print("=" * 60)
    
    try:
        # Run sync example
        sync_example()
        
        # Run async example
        asyncio.run(async_example())
        
        print("\n" + "=" * 60)
        print("✅ Examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the agent server is running on http://localhost:8000")


if __name__ == "__main__":
    main()
