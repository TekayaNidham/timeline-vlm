# Contributing to timeline-vlm

Thank you for your interest in contributing! This project is the official implementation of *"A Matter of Time: Revealing the Structure of Time in Vision-Language Models"* (ACM MM '25).

## Ways to Contribute

- **Bug reports** — open an issue using the bug report template
- **Feature requests** — open an issue using the feature request template
- **Model additions** — add support for new VLMs to the evaluation pipeline
- **Dataset contributions** — expand or improve the TIME10k benchmark
- **Documentation** — fix typos, clarify explanations, add examples
- **Code improvements** — performance, readability, test coverage

## Getting Started

1. Fork the repository and clone your fork.
2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Development Guidelines

- Keep changes focused — one logical change per PR.
- Follow the existing code style (PEP 8).
- Add or update docstrings for any public API changes.
- If adding a new model, place it under `timeline_vlm/models/` and document it in [docs/models.md](docs/models.md).
- Run the existing scripts on a small subset before submitting to verify nothing is broken.

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a PR against the `main` branch.
3. Fill in the PR template — describe what changed and why.
4. Link any related issues.

## Reporting Issues

Please use the issue templates. Include:
- Python version and OS
- `timeline-vlm` version (`pip show timeline-vlm`)
- Minimal reproduction steps or code snippet
- Full error traceback if applicable

## Code of Conduct

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
