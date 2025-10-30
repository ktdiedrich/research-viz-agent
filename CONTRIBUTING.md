# Contributing to Medical CV Research Agent

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/research-viz-agent.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`

## Development Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes (see Testing section)
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

Example:
```python
def search_papers(
    self,
    query: str,
    max_results: int = 10
) -> List[Dict]:
    """
    Search for papers.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        
    Returns:
        List of paper dictionaries
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_mcp_tools.py

# Run with coverage
python -m pytest --cov=research_viz_agent tests/
```

### Test Individual Tools

```bash
# Test MCP tools without OpenAI API
python tests/test_tools_standalone.py
```

### Writing Tests

- Add tests for new features
- Ensure existing tests pass
- Aim for good code coverage
- Use mocks for external API calls in unit tests

## Adding New Features

### Adding a New Research Source

1. Create a new tool in `research_viz_agent/mcp_tools/`
2. Follow the pattern of existing tools (ArxivTool, PubMedTool, etc.)
3. Implement:
   - `__init__()`: Initialize the tool
   - `search_papers()` or similar: Main search method
   - `search_medical_cv_models()`: Domain-specific search
4. Add the tool to the workflow in `research_workflow.py`
5. Update documentation

### Improving Summarization

1. Modify prompts in `research_workflow.py`
2. Adjust the `_prepare_context()` method to include more/different information
3. Test with various queries

### Adding CLI Options

1. Edit `research_viz_agent/cli.py`
2. Add new arguments to the argument parser
3. Pass arguments to the agent
4. Update documentation

## Documentation

- Update README.md for user-facing changes
- Update docs/API.md for API changes
- Update docs/CONFIGURATION.md for config changes
- Add examples for new features

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### PR Description

Include:
- What changes were made
- Why the changes were made
- Any breaking changes
- Related issues (if any)

## Bug Reports

When reporting bugs, include:

1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment (OS, Python version, etc.)
6. Error messages and stack traces

## Feature Requests

When requesting features, include:

1. Description of the feature
2. Use case / motivation
3. Proposed implementation (if you have ideas)
4. Any alternatives you've considered

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions

## Questions?

If you have questions:

1. Check existing documentation
2. Search existing issues
3. Open a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

Thank you for contributing! 🎉
