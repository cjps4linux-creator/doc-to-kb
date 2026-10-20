# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-20

### Added

- **Real CI test gate.** The workflow ran `pytest -q || echo "No tests yet"`,
  which executes no tests and always exits zero. It is replaced with a
  pinned `pytest==8.3.3` installation and an unconditional `pytest -q`.
  The suite has 50 tests.
- **Dependency pin guard** (`scripts/pin_guard.py`), wired as a required
  `pin-check` status check.

### Changed

- **Ruff rule set pinned** to `E4,E7,E9,F` with `ruff==0.17.1`.
- **README author credit** added: Conrad CJ Wilson.

## [0.1.0]

- Initial release: offline semantic search over local PDF collections

[1.0.0]: https://github.com/cjps4linux-creator/doc-to-kb/releases/tag/v1.0.0
