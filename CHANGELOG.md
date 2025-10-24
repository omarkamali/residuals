# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-24

### Added
- Package release on PyPi
- Support for Python 3.13 and 3.14

### Added
- Initial release of residuals package
- Core API is class-only: `Residuals` with `from_models`, `from_pretrained`, `apply`, `save_pretrained`
- Residual artifacts always include the instruct tokenizer saved alongside weights
- Implementation based on Samsung Research 2024 paper (Jindal et al.)
- Task arithmetic methodology following Ilharco et al. 2022
- Automatic tokenizer alignment with PAD token handling
- Comprehensive test suite with >90% coverage
- Full documentation with usage examples
- GitHub Actions CI/CD with PyPI Trusted Publisher
- Support for Python 3.8+

### Features
- Element-wise residual calculation (no scaling needed for same-family)
- Safe tokenizer alignment before residual application (requires both base and instruct tokenizers)
- Zero-initialization of new embedding rows
- Full type hints and docstrings with paper references
- Compatible with Unsloth, HuggingFace Transformers

[0.1.0]: https://github.com/omarkamali/residuals/releases/tag/v0.1.0
