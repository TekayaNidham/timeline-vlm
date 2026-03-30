# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✓         |
| Older   | ✗         |

Only the latest release on [PyPI](https://pypi.org/project/timeline-vlm/) receives security fixes.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

To report a vulnerability, open a [GitHub Security Advisory](https://github.com/Nidhamtek/timeline-vlm/security/advisories/new) (private disclosure). Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Any suggested mitigations

You can expect an acknowledgement within 72 hours and a resolution timeline within 7 days for confirmed issues.

## Scope

This project processes images and model embeddings locally. Key areas of concern include:
- Arbitrary code execution via malicious model files or configs
- Path traversal in file-loading utilities
- Dependency vulnerabilities (tracked via `requirements.txt` / `pyproject.toml`)
