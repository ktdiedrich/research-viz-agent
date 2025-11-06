#!/usr/bin/env python3
"""
Quick test to verify the serve command works.
"""
import subprocess
import sys
import time
import requests

def test_serve_command():
    """Test that the serve command starts successfully."""
    print("Testing 'research-viz-agent serve' command...")
    
    # Start the server in the background
    print("Starting server...")
    process = subprocess.Popen(
        ["uv", "run", "research-viz-agent", "serve", "--port", "9999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:9999/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server started successfully")
            print(f"✓ Health check passed: {response.json()}")
            
            # Test status endpoint
            response = requests.get("http://localhost:9999/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Status endpoint works: {status['agent_name']}")
                return True
            else:
                print(f"✗ Status endpoint failed: {response.status_code}")
                return False
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Server not responding: {e}")
        return False
        
    finally:
        # Stop the server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✓ Server stopped")


if __name__ == "__main__":
    success = test_serve_command()
    sys.exit(0 if success else 1)
