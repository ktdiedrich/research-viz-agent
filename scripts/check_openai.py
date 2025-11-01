#!/usr/bin/env python3
"""
Utility script to check OpenAI API setup and quota.
"""
import os
from dotenv import load_dotenv


def check_openai_setup():
    """Check OpenAI API key and basic connectivity."""
    print("="*60)
    print("OpenAI API Setup Checker")
    print("="*60)
    print()
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found")
        print()
        print("To fix this:")
        print("1. Get an API key from: https://platform.openai.com/api-keys")
        print("2. Add it to your .env file:")
        print("   echo 'OPENAI_API_KEY=your-key-here' >> .env")
        print("3. Or export it in your shell:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print()
        print("Alternatives:")
        print("- Use --no-summary flag to skip AI summarization")
        print("- Use --rag-search to search existing results")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Test API connectivity
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Make a minimal API call to test quota
        print("🔍 Testing API connectivity...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("✅ API connection successful!")
        print(f"✅ Model response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API test failed: {e}")
        print()
        
        if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
            print("💳 Quota Issue:")
            print("- Check billing: https://platform.openai.com/account/billing")
            print("- Add payment method or credits")
            print("- Upgrade from free tier if needed")
            print()
            print("Free alternatives:")
            print("- Use --no-summary to collect results without AI summarization")
            print("- Search existing RAG database with --rag-search")
            
        elif "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            print("🔑 Authentication Issue:")
            print("- Verify your API key is correct")
            print("- Generate a new key: https://platform.openai.com/api-keys")
            print("- Check for extra spaces or quotes in your .env file")
            
        elif "rate_limit" in error_msg.lower():
            print("⏱️ Rate Limit Issue:")
            print("- Wait a moment and try again")
            print("- Your quota is fine, just hitting rate limits")
            
        else:
            print("🌐 Connection Issue:")
            print("- Check your internet connection")
            print("- Verify OpenAI services status: https://status.openai.com/")
        
        return False


def main():
    """Main function."""
    success = check_openai_setup()
    
    print()
    print("="*60)
    if success:
        print("🎉 All systems ready! You can run:")
        print("   python -m research_viz_agent.cli 'your query'")
    else:
        print("⚠️ Issues found. Consider these options:")
        print("   # Research without AI summarization:")
        print("   python -m research_viz_agent.cli 'query' --no-summary")
        print()
        print("   # Search existing RAG database:")
        print("   python -m research_viz_agent.cli --rag-search 'query'")
        print()
        print("   # Check RAG database:")
        print("   python -m research_viz_agent.cli --rag-stats")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())