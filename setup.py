from setuptools import setup, find_packages

setup(
    name="research-viz-agent",
    version="0.1.0",
    description="AI agent for summarizing medical computer vision AI models from scientific research",
    author="ktdiedrich",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.20",
        "langgraph>=0.0.20",
        "mcp>=0.1.0",
        "arxiv>=2.0.0",
        "biopython>=1.83",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.9",
)
