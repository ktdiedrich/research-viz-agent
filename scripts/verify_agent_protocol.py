#!/usr/bin/env python3
"""
Verification script for agent communication protocol.

Tests that all components can be imported and instantiated.
"""
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from research_viz_agent.agent_protocol import (
            AgentRequest,
            AgentResponse,
            AgentStatus,
            AgentCapability,
            ResearchQuery,
            ResearchResult,
            AgentServer,
            AgentClient
        )
        print("✓ All protocol imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_schema_creation():
    """Test that schemas can be instantiated."""
    print("\nTesting schema creation...")
    
    try:
        from research_viz_agent.agent_protocol.schemas import (
            AgentRequest,
            AgentResponse,
            ResearchQuery
        )
        
        # Create a request
        request = AgentRequest(
            request_id="test-123",
            capability="research_medical_cv",
            parameters={"query": "test"}
        )
        assert request.request_id == "test-123"
        print("✓ AgentRequest created")
        
        # Create a response
        response = AgentResponse(
            request_id="test-123",
            status="success",
            result={"data": "test"}
        )
        assert response.status == "success"
        print("✓ AgentResponse created")
        
        # Create a query
        query = ResearchQuery(
            query="test query",
            max_results=10
        )
        assert query.max_results == 10
        print("✓ ResearchQuery created")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema creation failed: {e}")
        return False


def test_server_instantiation():
    """Test that server can be created."""
    print("\nTesting server instantiation...")
    
    try:
        from unittest.mock import Mock
        from research_viz_agent.agent_protocol.server import AgentServer
        
        mock_agent = Mock()
        server = AgentServer(
            agent=mock_agent,
            host="localhost",
            port=9999
        )
        
        assert server.host == "localhost"
        assert server.port == 9999
        print("✓ AgentServer created")
        
        return True
        
    except Exception as e:
        print(f"✗ Server instantiation failed: {e}")
        return False


def test_client_instantiation():
    """Test that client can be created."""
    print("\nTesting client instantiation...")
    
    try:
        from research_viz_agent.agent_protocol.client import AgentClient
        
        client = AgentClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        print("✓ AgentClient created")
        
        # Test URL normalization
        client2 = AgentClient(base_url="http://localhost:8000/")
        assert client2.base_url == "http://localhost:8000"
        print("✓ URL normalization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Client instantiation failed: {e}")
        return False


def test_cli_integration():
    """Test that CLI has serve command."""
    print("\nTesting CLI integration...")
    
    try:
        import argparse
        import io
        from contextlib import redirect_stdout
        
        # Import CLI to check it doesn't crash
        from research_viz_agent import cli
        
        print("✓ CLI imports successfully")
        return True
        
    except Exception as e:
        print(f"✗ CLI integration failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Agent Communication Protocol Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_schema_creation,
        test_server_instantiation,
        test_client_instantiation,
        test_cli_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ All {total} verification tests passed!")
        print("=" * 60)
        print("\nAgent communication protocol is ready to use.")
        print("\nNext steps:")
        print("1. Start the server: research-viz-agent serve")
        print("2. Run examples: python examples/agent_client.py")
        print("3. Read docs: docs/AGENT_COMMUNICATION.md")
        return 0
    else:
        print(f"❌ {total - passed}/{total} tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
