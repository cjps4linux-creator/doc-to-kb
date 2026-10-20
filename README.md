<h1 align="center">DocToKB</h1>

<p align="center"><b>Local offline semantic search over your PDF collection. No cloud, no API keys, no upload.</b></p>

<p align="center">
  <a href="https://github.com/cjps4linux-creator/doc-to-kb/actions/workflows/ci.yml"><img src="https://github.com/cjps4linux-creator/doc-to-kb/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/cjps4linux-creator/doc-to-kb/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/release-v1.0.0-0a7563" alt="Release v1.0.0"></a>
  <img src="https://img.shields.io/badge/license-MIT-0a7563" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776ab" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-8a5800" alt="Platforms">
  <img src="https://img.shields.io/badge/no_account-no_telemetry-0a7563" alt="No account, no telemetry">
</p>

---

## What it does

Point DocToKB at a folder of PDFs. It chunks every document, embeds the
chunks locally with `sentence-transformers`, and answers natural-language
queries against them.

- **Fully offline** — everything runs on your machine
- **Local embeddings** — `all-MiniLM-L6-v2`, downloaded once on first run
- **Semantic search** — query in plain language, get ranked passages
- **Lightweight GUI** — Tkinter, no browser required

## Requirements

- Python 3.11+
- ~2 GB RAM while the embedding model is loaded

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Or double-click `DocToKB.bat` on the Desktop.

1. **Choose PDF Folder**
2. **Index Folder** — extracts chunks and builds embeddings (first run downloads
   the model, so it takes a moment)
3. Type a natural-language query and click **Search** — the top 5 matches are
   shown with filename, similarity score and snippet

## Notes

- PDFs only at present
- The index is held in memory for the session and rebuilt on restart

## Packaging

A single-file Windows build is produced with PyInstaller:

```bash
pip install pyinstaller
pyinstaller DocToKB.spec
```

The `build/` and `dist/` directories are intentionally not committed.

## Licence

MIT — see [LICENSE](LICENSE).

---

## Credits

Built by **Conrad CJ Wilson** — [GitHub](https://github.com/cjps4linux-creator) · [LinkedIn](https://linkedin.com/in/conradcjwilson)

`conradcjwilson0@gmail.com`
