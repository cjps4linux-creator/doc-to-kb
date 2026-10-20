"""Shared pytest fixtures for the doc-to-kb test suite.

Responsibilities:

* Put the repository root on ``sys.path`` so ``import app`` resolves without an
  editable install (pytest only prepends the *tests* directory by default).
* Provide a pure-Python stand-in for the tkinter widget the application uses,
  so ``App.extract_pdf`` can be exercised without a running Tk event loop.
* Provide a tiny PDF writer so the tests can build real (but tiny) PDFs at
  runtime instead of shipping a binary corpus.

The suite is deliberately hermetic: no network access, no sentence-transformers
model download, no tkinter root window, and no files written outside of
pytest's ``tmp_path``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (the directory holding ``app.py``)."""
    return REPO_ROOT


# --------------------------------------------------------------------------- #
# Tkinter stand-ins
# --------------------------------------------------------------------------- #
class FakeTextWidget:
    """Minimal stand-in for ``tkinter.Text`` that records ``insert`` calls.

    ``App.extract_pdf`` reports whole-file failures by calling
    ``self.text.insert(tk.END, ...)``; recording those calls lets the tests
    assert on the error path without instantiating any widget.
    """

    def __init__(self) -> None:
        self.inserted: list[tuple[object, str]] = []

    def insert(self, index: object = "end", chars: str = "", *args: object) -> None:
        self.inserted.append((index, chars))

    @property
    def text(self) -> str:
        """Everything that was inserted, concatenated in insertion order."""
        return "".join(chars for _index, chars in self.inserted)

    @property
    def call_count(self) -> int:
        return len(self.inserted)


class DummyApp:
    """Stand-in for ``app.App`` exposing only what the tested methods touch.

    ``chunk_text`` does not use ``self`` at all, so it is safe to call the
    unbound method with ``None``.  ``extract_pdf`` needs a ``.text`` widget for
    its error path, which is what this class provides.
    """

    def __init__(self) -> None:
        self.text = FakeTextWidget()


@pytest.fixture()
def dummy_app() -> DummyApp:
    """A fresh stand-in for ``app.App`` with a recording text widget."""
    return DummyApp()


# --------------------------------------------------------------------------- #
# Safety guard: the suite must never trigger a model download
# --------------------------------------------------------------------------- #
class ForbiddenSentenceTransformer:
    """Placeholder that fails loudly if a test tries to build the model."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Tests must not construct a SentenceTransformer "
            "(that would download the all-MiniLM-L6-v2 model)."
        )


@pytest.fixture(autouse=True)
def forbid_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``app.SentenceTransformer`` with a tripwire for every test.

    ``app.py`` binds ``SentenceTransformer`` at import time, so patching the
    module attribute is enough to guarantee that no test (including a future
    one that mistakenly instantiates ``App``) can trigger a model download.
    """
    module = sys.modules.get("app")
    if module is not None:
        monkeypatch.setattr(module, "SentenceTransformer", ForbiddenSentenceTransformer, raising=False)


# --------------------------------------------------------------------------- #
# Tiny PDF writer
# --------------------------------------------------------------------------- #
def _escape_pdf_literal(text: str) -> bytes:
    """Escape a string for use inside a PDF ``(...)`` literal."""
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return out.encode("latin-1", "replace")


def build_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a minimal, valid PDF whose pages render the given ASCII strings.

    Uses raw PDF syntax rather than ``pypdf.PdfWriter`` because the writer
    cannot emit page content streams.  The result is parsed back by
    ``pypdf.PdfReader`` and its text is extracted normally, which is exactly
    what ``App.extract_pdf`` does.
    """
    if not page_texts:
        raise ValueError("build_pdf_bytes() needs at least one page")

    objects: dict[int, bytes] = {}
    page_count = len(page_texts)
    kids = " ".join(f"{3 + i} 0 R" for i in range(page_count))
    font_id = 3 + page_count
    content_start = font_id + 1

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode()
    for i in range(page_count):
        page_id = 3 + i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_start + i} 0 R >>"
        ).encode()
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, page_text in enumerate(page_texts):
        stream = b"BT /F1 12 Tf 72 720 Td (" + _escape_pdf_literal(page_text) + b") Tj ET"
        content_id = content_start + i
        objects[content_id] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = out.tell()
        out.write(f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n")

    xref_position = out.tell()
    size = max(objects) + 1
    out.write(f"xref\n0 {size}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for number in range(1, size):
        out.write(f"{offsets[number]:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n".encode())
    return out.getvalue()


@pytest.fixture()
def pdf_factory(tmp_path: Path):
    """Return a callable writing a small PDF into ``tmp_path`` and returning its path."""

    def make_pdf(page_texts: list[str], name: str = "sample.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(build_pdf_bytes(page_texts))
        return path

    return make_pdf
