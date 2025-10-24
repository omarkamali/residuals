# Contributing to Residuals

Thank you for your interest in contributing to the Residuals package! This document provides guidelines and instructions for contributing.

## Development Setup

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/omarkamali/residuals.git
cd residuals

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install with dev dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
git clone https://github.com/omarkamali/residuals.git
cd residuals
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=residuals --cov-report=html

# Run specific test
uv run pytest tests/test_residuals.py::test_calculate_and_apply_residuals
```

## Code Quality

We use modern tooling for code quality:

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type checking (optional)
uv run mypy src/residuals
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Add tests** for any new functionality
3. **Update documentation** if you change APIs
4. **Ensure tests pass** and code is formatted
5. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No lint errors (`ruff check`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated with your changes

## Commit Messages

We optionally follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example: `feat: add support for cross-family residual scaling`

## Release Process

Releases are automated via GitHub Actions with Trusted Publishers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a GitHub release with tag `vX.Y.Z`
4. GitHub Actions will build and publish to PyPI automatically

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## Questions?

Open an issue or reach out to residuals@omarkama.li

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
