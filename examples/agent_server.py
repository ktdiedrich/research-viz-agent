"""
Example: Agent Server

This example shows how to start the research agent as an HTTP server
that other agents can communicate with.

To run this example:
    python examples/agent_server.py

Or use the CLI:
    research-viz-agent serve
    research-viz-agent serve --port 8080 --host 0.0.0.0
"""
from research_viz_agent.agent_protocol.server import create_agent_server


def main():
    """Start the agent server."""
    # Create and configure the server
    server = create_agent_server(
        llm_provider="github",  # Use GitHub Models (free)
        host="0.0.0.0",         # Listen on all interfaces
        port=8000,              # Port 8000
        enable_rag=True,        # Enable RAG storage
        max_results=20          # Max results per source
    )
    
    print("🤖 Starting Medical CV Research Agent Server")
    print("   Base URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Status: http://localhost:8000/status")
    print("\nPress Ctrl+C to stop\n")
    
    # Run the server (blocks until stopped)
    server.run()


if __name__ == "__main__":
    main()
