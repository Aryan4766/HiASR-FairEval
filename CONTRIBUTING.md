# Contributing to HiASR-FairEval

Thank you for your interest in contributing to the Hindi ASR Fair Evaluation project!

## How to Contribute

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and add tests
4. **Ensure tests pass**: `pytest tests/ -v`
5. **Commit** with clear messages: `git commit -m "Add feature description"`
6. **Push** and open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/HiASR-FairEval.git
cd HiASR-FairEval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements-dev.txt
```

## Code Style

- Follow **PEP 8** conventions
- Use **type hints** on all function signatures
- Add **docstrings** (Google-style) to all public functions
- Format code: `black src/ tests/`
- Lint: `flake8 src/ tests/`

## Testing

All new code must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Pull Request Guidelines

- Reference related issues (e.g., `Fixes #12`)
- Include a description of what changed and why
- Ensure all CI checks pass
- Keep PRs focused — one feature/fix per PR

## Reporting Issues

Use GitHub Issues with:

- Clear, descriptive title
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Python version and OS

## Questions?

Open a discussion or issue — we're happy to help!
