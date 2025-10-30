# Quick Start Guide

Get started with the Medical CV Research Agent in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/ktdiedrich/research-viz-agent.git
cd research-viz-agent

# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## 2. Configuration

Set up your OpenAI API key:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

## 3. Run Your First Query

### Option A: Command Line

```bash
python -m research_viz_agent.cli "lung cancer detection" --email your@email.com
```

### Option B: Python Script

Create a file `my_research.py`:

```python
from research_viz_agent.agents.medical_cv_agent import MedicalCVResearchAgent

# Initialize the agent
agent = MedicalCVResearchAgent(
    pubmed_email="your-email@example.com"
)

# Run research
results = agent.research("lung cancer detection")

# Print results
print(agent.format_results(results))
```

Run it:
```bash
python my_research.py
```

## 4. Example Queries

Try these example queries:

- `"lung cancer detection"`
- `"skin lesion classification"`
- `"brain tumor segmentation"`
- `"diabetic retinopathy detection"`
- `"chest x-ray analysis"`
- `"histopathology image analysis"`

## 5. Save Results

### Command Line
```bash
python -m research_viz_agent.cli "brain tumor segmentation" --output results.txt
```

### Python
```python
results = agent.research("brain tumor segmentation")
formatted = agent.format_results(results)

with open("results.txt", "w") as f:
    f.write(formatted)
```

## 6. Access Individual Results

```python
results = agent.research("lung cancer detection")

# Get summary
print(results['summary'])

# Access papers
for paper in results['arxiv_results'][:5]:
    print(f"- {paper['title']}")
    print(f"  URL: {paper['pdf_url']}")

# Access models
for model in results['huggingface_results'][:5]:
    print(f"- {model['model_id']}")
    print(f"  Downloads: {model['downloads']}")

# Get counts
print(f"Total papers: {results['total_papers']}")
print(f"Total models: {results['total_models']}")
```

## 7. Customize the Agent

### Use GPT-4
```python
agent = MedicalCVResearchAgent(
    pubmed_email="your@email.com",
    model_name="gpt-4"  # More capable but slower
)
```

### Adjust Temperature
```python
agent = MedicalCVResearchAgent(
    pubmed_email="your@email.com",
    temperature=0.3  # More focused (0.0-1.0)
)
```

## 8. Test Without OpenAI API

Test the individual MCP tools without needing an OpenAI API key:

```bash
python tests/test_tools_standalone.py
```

This will test:
- ✓ arXiv search
- ✓ PubMed search
- ✓ HuggingFace model search

## Troubleshooting

### "OpenAI API key not found"
- Make sure `.env` file exists with `OPENAI_API_KEY=your-key`
- Or set it in your environment: `export OPENAI_API_KEY=your-key`

### "No results found"
- Try a broader search query
- Check your internet connection
- Verify the research sources are accessible

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific packages
pip install langchain langchain-openai langgraph arxiv biopython
```

## Next Steps

- Read the full [README.md](../README.md)
- Check the [API Documentation](API.md)
- See [Configuration Guide](CONFIGURATION.md) for advanced options
- Contribute! See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Example Output

```
============================================================
RESEARCH SUMMARY: lung cancer detection
============================================================

Total Papers Found: 30
Total Models Found: 15

------------------------------------------------------------
AI-GENERATED SUMMARY
------------------------------------------------------------

The research reveals several state-of-the-art AI models for lung 
cancer detection using computer vision...

[Comprehensive analysis continues...]

------------------------------------------------------------
DETAILED SOURCES
------------------------------------------------------------

### ArXiv Papers ###

1. Deep Learning for Lung Nodule Detection in CT Scans
   URL: https://arxiv.org/abs/2301.12345
   Published: 2023-05-15

[Additional papers and models...]
```

## Support

For help:
- Open an issue: https://github.com/ktdiedrich/research-viz-agent/issues
- Read the documentation in the `docs/` folder
- Check existing issues for similar problems

Happy researching! 🔬🤖
