# Suggested Commands for Maou Project

## Package Management (uv only, NO pip)
```bash
uv sync                                    # Base install
uv sync --extra cpu                        # CPU PyTorch
uv sync --extra cuda                       # CUDA PyTorch
uv sync --extra cpu --extra visualize      # Testing with visualization
uv sync --extra cuda --extra visualize     # Full development
uv add <package>                           # Add new package
uv run <script>                            # Run script
```

## Quality Assurance Pipeline (run before commit)
```bash
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/
```

## Testing
```bash
uv run pytest                              # Run all tests
uv run pytest tests/path/to/test.py        # Run specific test
```

## Rust Extension
```bash
uv run maturin develop                     # Build Rust extension
```

## CLI
```bash
uv run maou --help                         # CLI help
```

## Git Commit Format
```
feat|fix|docs|refactor|test|perf: message
```
