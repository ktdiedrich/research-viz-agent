"""
Example usage of the Medical CV Research Agent.
"""
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent


def main():
    """Run example research queries."""
    
    # Initialize the agent
    # Make sure to set OPENAI_API_KEY in your .env file
    agent = MedicalCVResearchAgent(
        pubmed_email="your-email@example.com",  # Change this to your email
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Example queries
    queries = [
        "lung cancer detection",
        "skin lesion classification",
        "brain tumor segmentation"
    ]
    
    # Run a single query
    query = queries[0]
    print(f"\nRunning example query: {query}\n")
    
    results = agent.research(query)
    formatted_output = agent.format_results(results)
    
    print(formatted_output)
    
    # Optionally save to file
    with open("research_output.txt", "w", encoding="utf-8") as f:
        f.write(formatted_output)
    
    print("\n\nResults saved to research_output.txt")


if __name__ == "__main__":
    main()
