# CONTRIBUTING

Thank you for your interest in contributing to AML Bandits!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit with clear messages: `git commit -m "feat: add new algorithm"`
7. Push and create a Pull Request

## Development Setup

```bash
# Clone and install
git clone https://github.com/Cataldir/aml-foundry-integration.git
cd aml-foundry-integration
pip install -e ".[dev,all]"

# Run tests
pytest tests/

# Run code quality checks
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Code Style

- **Formatting**: Black (100 char line length)
- **Linting**: Ruff
- **Type hints**: Preferred but not strict
- **Docstrings**: Google-style docstrings

## Testing

- Add tests for any new features in `tests/`
- Run: `pytest -xvs`
- Coverage: `pytest --cov=src/aml_bandits`

## Pull Request Guidelines

- Clear title and description of changes
- Reference any related issues
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## Reporting Issues

Please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages/tracebacks

## Questions?

Open an issue or reach out to the maintainers.
