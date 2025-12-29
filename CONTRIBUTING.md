# Contributing to cfDNA-Gen

Thank you for your interest in contributing to cfDNA-Gen! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cfdna-gen.git
   cd cfdna-gen
   ```

3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Type Checking

```bash
mypy cfdna_gen/
```

## Code Style Guidelines

- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for public functions (Google style)
- Keep functions focused and small
- Add tests for new functionality

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add a clear description of changes
4. Reference any related issues

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Minimal reproducible example
- Full error traceback

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or discussions about the project.
