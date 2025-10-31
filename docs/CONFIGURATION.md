# Configuration Guide

## Environment Variables

The agent requires certain environment variables to be configured. Create a `.env` file in the project root based on `.env.example`.

### Required

- **OPENAI_API_KEY**: Your OpenAI API key
  - Get it from: https://platform.openai.com/api-keys
  - Example: `sk-...`

### Optional

- **HUGGINGFACE_TOKEN**: Your HuggingFace API token
  - Get it from: https://huggingface.co/settings/tokens
  - Used for authenticated HuggingFace API requests
  - Not required for basic functionality

## Agent Configuration

### Model Selection

You can choose different OpenAI models based on your needs:

- **gpt-3.5-turbo** (default): Fast and cost-effective
- **gpt-4**: More capable but slower and more expensive
- **gpt-4-turbo**: Balance of capability and speed

### Temperature

Controls the creativity/randomness of responses:

- **0.0**: Deterministic, focused responses
- **0.7** (default): Balanced creativity
- **1.0**: More creative and varied responses

### PubMed Email

Required by NCBI for PubMed API access:
- Use your actual email address
- NCBI may contact you if there are issues with your requests
- Default: `research@example.com` (change this!)

## API Rate Limits

### arXiv
- No authentication required
- Rate limit: ~3 requests per second
- The tool automatically handles pagination

### PubMed
- No authentication required
- Rate limit: 3 requests per second (without API key)
- 10 requests per second (with API key)
- The tool includes automatic delays to respect limits

### HuggingFace
- Public API: No authentication required for basic searches
- Authenticated: Higher rate limits with token
- Rate limit: Varies by endpoint

### OpenAI
- Requires API key
- Rate limits vary by model and tier
- See: https://platform.openai.com/docs/guides/rate-limits

## Example Configurations

### Development/Testing
```python
agent = MedicalCVResearchAgent(
    pubmed_email="your-email@example.com",
    model_name="gpt-3.5-turbo",
    temperature=0.5
)
```

### Production (High Quality)
```python
agent = MedicalCVResearchAgent(
    pubmed_email="your-email@example.com",
    model_name="gpt-4",
    temperature=0.3
)
```

### Cost-Optimized
```python
agent = MedicalCVResearchAgent(
    pubmed_email="your-email@example.com",
    model_name="gpt-3.5-turbo",
    temperature=0.7
)
```

## Troubleshooting

### API Key Issues
- Ensure `OPENAI_API_KEY` is set in `.env` file
- Verify the key is valid and has credits
- Check that `.env` is in the same directory as your script

### Rate Limit Errors
- Wait a few seconds and retry
- Reduce the number of concurrent requests
- Consider using API keys for services that support them

### No Results Found
- Try broader search terms
- Check your internet connection
- Verify the research sources are accessible

### Import Errors
- Ensure all dependencies are installed: `poetry install` or `pip install -e .`
- Check Python version is 3.9+
- Try reinstalling in a fresh virtual environment
