"""
Example: Agent-to-Agent Communication

This example demonstrates how two agents communicate with each other.

Scenario: An orchestrator agent delegates research tasks to the
medical CV research agent and aggregates results.
"""
import asyncio
from typing import List, Dict, Any
from research_viz_agent.agent_protocol.client import AgentClient


class OrchestratorAgent:
    """
    Example orchestrator agent that coordinates multiple research tasks.
    
    This demonstrates how an external agent can use the research agent
    as a service to gather information.
    """
    
    def __init__(self, research_agent_url: str = "http://localhost:8000"):
        """Initialize the orchestrator with a connection to the research agent."""
        self.research_client = AgentClient(base_url=research_agent_url)
        self.results = []
    
    async def research_multiple_topics(
        self,
        topics: List[str],
        max_results_per_topic: int = 10
    ) -> Dict[str, Any]:
        """
        Research multiple topics concurrently using the research agent.
        
        Args:
            topics: List of research topics
            max_results_per_topic: Max results per topic
        
        Returns:
            Aggregated results from all topics
        """
        print(f"🔬 Orchestrator: Researching {len(topics)} topics concurrently...\n")
        
        # Create concurrent research tasks
        tasks = [
            self.research_client.research_async(
                query=topic,
                max_results=max_results_per_topic,
                enable_rag=True
            )
            for topic in topics
        ]
        
        # Execute all research tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        total_papers = sum(r.total_papers for r in results)
        total_models = sum(r.total_models for r in results)
        
        aggregated = {
            "topics": topics,
            "total_topics": len(topics),
            "total_papers": total_papers,
            "total_models": total_models,
            "results_by_topic": {
                topic: {
                    "papers": result.total_papers,
                    "models": result.total_models,
                    "summary": result.summary
                }
                for topic, result in zip(topics, results)
            }
        }
        
        return aggregated
    
    async def find_related_work(
        self,
        seed_query: str,
        num_expansions: int = 2
    ) -> Dict[str, Any]:
        """
        Use RAG to iteratively find related work.
        
        1. Initial research on seed query
        2. Search RAG for related topics
        3. Research those topics
        
        Args:
            seed_query: Initial research query
            num_expansions: How many related topics to explore
        
        Returns:
            Results from seed query and related work
        """
        print(f"🔍 Orchestrator: Finding related work for '{seed_query}'...\n")
        
        # Step 1: Initial research
        print("  Step 1: Initial research...")
        initial_result = await self.research_client.research_async(
            query=seed_query,
            max_results=20,
            enable_rag=True
        )
        print(f"    Found {initial_result.total_papers} papers\n")
        
        # Step 2: Find related topics from RAG
        print("  Step 2: Searching RAG for related work...")
        rag_results = await self.research_client.search_rag_async(
            query=seed_query,
            k=num_expansions * 3
        )
        print(f"    Found {rag_results.total_count} related documents\n")
        
        # Extract unique topics from RAG results (simplified)
        # In practice, you'd use more sophisticated topic extraction
        related_topics = []
        for doc in rag_results.results[:num_expansions]:
            # Use the title or abstract as a new query
            if 'title' in doc:
                related_topics.append(doc['title'][:50])  # Truncate
        
        # Step 3: Research related topics
        if related_topics:
            print(f"  Step 3: Researching {len(related_topics)} related topics...")
            related_results = await self.research_multiple_topics(
                topics=related_topics,
                max_results_per_topic=5
            )
        else:
            related_results = None
        
        return {
            "seed_query": seed_query,
            "initial_papers": initial_result.total_papers,
            "initial_summary": initial_result.summary,
            "related_work": related_results
        }
    
    async def close(self):
        """Close the client connection."""
        await self.research_client.close_async()


async def main():
    """Run the orchestrator agent examples."""
    print("=" * 70)
    print("Agent-to-Agent Communication Example")
    print("=" * 70)
    print("\nOrchestrator Agent coordinating with Research Agent\n")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = OrchestratorAgent(research_agent_url="http://localhost:8000")
    
    try:
        # Example 1: Research multiple topics concurrently
        print("\n📋 Example 1: Multi-Topic Research\n")
        topics = [
            "lung cancer detection CNN",
            "brain tumor segmentation U-Net",
            "retinal disease classification"
        ]
        
        multi_results = await orchestrator.research_multiple_topics(
            topics=topics,
            max_results_per_topic=5
        )
        
        print(f"✅ Results:")
        print(f"   Total papers found: {multi_results['total_papers']}")
        print(f"   Total models found: {multi_results['total_models']}")
        for topic, data in multi_results['results_by_topic'].items():
            print(f"\n   Topic: {topic}")
            print(f"   - Papers: {data['papers']}, Models: {data['models']}")
            print(f"   - Summary: {data['summary'][:100]}...")
        
        # Example 2: Iterative research with RAG
        print("\n" + "=" * 70)
        print("\n📋 Example 2: Iterative Research with RAG\n")
        
        related_work = await orchestrator.find_related_work(
            seed_query="deep learning medical imaging",
            num_expansions=2
        )
        
        print(f"✅ Results:")
        print(f"   Seed query: {related_work['seed_query']}")
        print(f"   Initial papers: {related_work['initial_papers']}")
        print(f"   Summary: {related_work['initial_summary'][:150]}...")
        
        if related_work['related_work']:
            print(f"\n   Related work explored:")
            print(f"   - Total papers: {related_work['related_work']['total_papers']}")
            print(f"   - Topics: {related_work['related_work']['topics']}")
        
        print("\n" + "=" * 70)
        print("✅ Agent-to-agent communication successful!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the research agent server is running:")
        print("  python examples/agent_server.py")
        
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    print("\nMake sure the agent server is running:")
    print("  python examples/agent_server.py\n")
    
    asyncio.run(main())
