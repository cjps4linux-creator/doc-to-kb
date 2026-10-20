"""Pytest suite for the doc-to-kb application.

Covers the two pieces of real, testable logic that live in ``app.py``:

* ``App.chunk_text(text, size=400)`` -- splits text into space-joined word
  chunks.  It does not touch ``self``, so it is exercised as an unbound method
  with a dummy ``self``; no ``App`` instance (and therefore no tkinter root and
  no sentence-transformers model) is ever created.
* ``App.extract_pdf(path)`` -- reads a PDF with ``pypdf`` and returns the
  extracted text, reporting whole-file failures through ``self.text.insert``.
  It is exercised with a dummy object that provides a recording ``.text``
  stand-in for the tkinter widget.

Everything here is offline and hermetic: no network, no model download, no
GUI, and no binary fixtures (the PDFs are generated at runtime).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import app


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_app_module_imports_without_building_the_model() -> None:
    """Importing ``app`` must not construct a SentenceTransformer."""
    module = sys.modules["app"]
    assert hasattr(module, "App")
    # The tripwire installed by the autouse fixture is in place: constructing
    # the real model would now raise, so the import was genuinely side-effect
    # free with respect to model download.
    assert module.SentenceTransformer.__name__ == "ForbiddenSentenceTransformer"


def test_app_instance_is_never_constructed(dummy_app) -> None:
    """The dummy stand-in is used instead of a real ``App``."""
    assert not isinstance(dummy_app, app.App)
    assert type(dummy_app).__name__ == "DummyApp"


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #
def chunk(text: str, size: int = 400) -> list[str]:
    """Call ``App.chunk_text`` without constructing an ``App``."""
    return list(app.App.chunk_text(None, text, size=size))


def test_chunk_text_is_a_generator() -> None:
    result = app.App.chunk_text(None, "a b c")
    assert hasattr(result, "__next__")
    assert next(result) == "a b c"


def test_chunk_text_basic_split() -> None:
    assert chunk("a b c d", size=2) == ["a b", "c d"]


def test_chunk_text_default_size_is_400() -> None:
    text = " ".join(str(i) for i in range(1000))
    chunks = list(app.App.chunk_text(None, text))
    assert len(chunks) == 3  # 400 + 400 + 200
    assert all(len(part.split()) <= 400 for part in chunks)
    assert chunks[0].split()[0] == "0"
    assert chunks[-1].split()[-1] == "999"


@pytest.mark.parametrize(
    ("text", "size", "expected"),
    [
        pytest.param("", 2, [], id="empty-string"),
        pytest.param("   ", 2, [], id="whitespace-only"),
        pytest.param("\n\t  \n", 4, [], id="newlines-and-tabs-only"),
        pytest.param("single", 10, ["single"], id="fewer-words-than-size"),
        pytest.param("a b c", 3, ["a b c"], id="exactly-equal-to-size"),
        pytest.param("a b c d", 2, ["a b", "c d"], id="exact-multiple-of-size"),
        pytest.param("a b c d e f", 3, ["a b c", "d e f"], id="multiple-of-three"),
        pytest.param("a b c", 1, ["a", "b", "c"], id="size-of-one"),
        pytest.param("a", 1, ["a"], id="size-of-one-single-word"),
        pytest.param("a b c", 1000, ["a b c"], id="very-large-size"),
        pytest.param("one", 400, ["one"], id="default-size-single-word"),
    ],
)
def test_chunk_text_sizes_and_edge_cases(text: str, size: int, expected: list[str]) -> None:
    assert chunk(text, size=size) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("a  b", ["a b"], id="double-space"),
        pytest.param("a\tb", ["a b"], id="tab-separated"),
        pytest.param("a\nb", ["a b"], id="newline-separated"),
        pytest.param("  leading and trailing  ", ["leading and trailing"], id="leading-trailing-space"),
        pytest.param("a \t\n b", ["a b"], id="mixed-whitespace-run"),
        pytest.param("a     b     c", ["a b c"], id="many-spaces-between-words"),
    ],
)
def test_chunk_text_normalises_whitespace(text: str, expected: list[str]) -> None:
    """``str.split()`` collapses runs of whitespace, so chunks are space-joined."""
    assert chunk(text, size=10) == expected


def test_chunk_text_preserves_word_order() -> None:
    words = [f"w{i}" for i in range(250)]
    chunks = chunk(" ".join(words), size=40)
    assert len(chunks) == 7  # 6 * 40 + 10
    assert " ".join(chunks).split() == words


def test_chunk_text_joins_with_single_spaces() -> None:
    for part in chunk("alpha\tbeta\ngamma  delta", size=2):
        assert "  " not in part
        assert "\t" not in part
        assert "\n" not in part


def test_chunk_text_never_exceeds_size_in_words() -> None:
    words = [f"w{i}" for i in range(1234)]
    for part in chunk(" ".join(words), size=100):
        assert len(part.split()) <= 100


def test_chunk_text_unicode() -> None:
    text = "café naïve 日本語 emoji 🚀"
    assert chunk(text, size=2) == ["café naïve", "日本語 emoji", "🚀"]


def test_chunk_text_multibyte_text_is_never_split_mid_character() -> None:
    """Non-ASCII runs without whitespace count as a single word.

    ``str.split()`` splits on whitespace only, so a run of CJK characters is
    one "word" and is never broken across chunk boundaries mid-character.
    """
    text = "日本語のテキスト"
    parts = chunk(text, size=3)
    assert parts == ["日本語のテキスト"]
    assert "".join(parts) == text


def test_chunk_text_multibyte_words_with_separators() -> None:
    """CJK separated by whitespace splits into words like any other text."""
    text = "日本語 テキスト ほか"
    assert chunk(text, size=2) == ["日本語 テキスト", "ほか"]


def test_chunk_text_unicode_only() -> None:
    """Emoji without whitespace form one word (they are not whitespace)."""
    assert chunk("🚀🎉🌍", size=2) == ["🚀🎉🌍"]


def test_chunk_text_unicode_with_separators() -> None:
    assert chunk("🚀 🎉 🌍", size=2) == ["🚀 🎉", "🌍"]


def test_chunk_text_large_input() -> None:
    words = [f"word{i}" for i in range(10_000)]
    chunks = chunk(" ".join(words), size=400)
    assert len(chunks) == 25
    assert " ".join(chunks).split() == words


def test_chunk_text_is_lazy() -> None:
    """Consuming one chunk must not require materialising the whole text."""
    generator = app.App.chunk_text(None, "a b c d", size=2)
    assert next(generator) == "a b"
    assert next(generator) == "c d"
    with pytest.raises(StopIteration):
        next(generator)


def test_chunk_text_result_is_independent_of_self() -> None:
    """``chunk_text`` never touches ``self``; two different dummies agree."""
    class Other:
        pass

    with_other = list(app.App.chunk_text(Other(), "x y z", size=2))
    with_none = list(app.App.chunk_text(None, "x y z", size=2))
    assert with_other == with_none == ["x y", "z"]


# --------------------------------------------------------------------------- #
# extract_pdf
# --------------------------------------------------------------------------- #
def test_extract_pdf_valid_pdf_returns_text(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["Hello World"])
    result = app.App.extract_pdf(dummy_app, str(path))
    assert "Hello World" in result


def test_extract_pdf_valid_pdf_does_not_report_errors(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["Hello World"])
    app.App.extract_pdf(dummy_app, str(path))
    assert dummy_app.text.call_count == 0


def test_extract_pdf_multiple_pages_are_joined(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["Page One", "Page Two", "Page Three"])
    result = app.App.extract_pdf(dummy_app, str(path))
    assert "Page One" in result
    assert "Page Two" in result
    assert "Page Three" in result
    assert result.count("\n") == 2  # pages joined with a newline


def test_extract_pdf_single_page_has_no_trailing_newline(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["Only Page"])
    result = app.App.extract_pdf(dummy_app, str(path))
    assert result == "Only Page"


def test_extract_pdf_non_pdf_file_returns_empty_string(dummy_app, tmp_path) -> None:
    path = tmp_path / "not-a-pdf.pdf"
    path.write_bytes(b"this is definitely not a pdf file")
    result = app.App.extract_pdf(dummy_app, str(path))
    assert result == ""


def test_extract_pdf_non_pdf_file_does_not_raise(dummy_app, tmp_path) -> None:
    path = tmp_path / "not-a-pdf.pdf"
    path.write_bytes(b"still not a pdf")
    try:
        app.App.extract_pdf(dummy_app, str(path))
    except Exception as exc:  # pragma: no cover - the call must not raise
        pytest.fail(f"extract_pdf raised on a non-PDF file: {exc!r}")


def test_extract_pdf_non_pdf_file_reports_error_to_text_widget(dummy_app, tmp_path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf at all")
    app.App.extract_pdf(dummy_app, str(path))
    assert dummy_app.text.call_count == 1
    index, message = dummy_app.text.inserted[0]
    assert index == "end"
    assert "broken.pdf" in message
    assert message.startswith("  err ")


def test_extract_pdf_empty_file_returns_empty_string(dummy_app, tmp_path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    assert app.App.extract_pdf(dummy_app, str(path)) == ""


def test_extract_pdf_truncated_pdf_is_handled(dummy_app, tmp_path) -> None:
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n")
    result = app.App.extract_pdf(dummy_app, str(path))
    assert isinstance(result, str)


def test_extract_pdf_missing_file_returns_empty_string(dummy_app, tmp_path) -> None:
    path = tmp_path / "does-not-exist.pdf"
    result = app.App.extract_pdf(dummy_app, str(path))
    assert result == ""
    assert dummy_app.text.call_count == 1
    assert "does-not-exist.pdf" in dummy_app.text.text


def test_extract_pdf_accepts_path_objects(dummy_app, pdf_factory) -> None:
    """``PdfReader`` accepts ``str`` and ``os.PathLike`` alike."""
    path = pdf_factory(["Path Object"])
    result = app.App.extract_pdf(dummy_app, path)
    assert "Path Object" in result


def test_extract_pdf_empty_page_yields_empty_segment(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["", "Second"])
    result = app.App.extract_pdf(dummy_app, str(path))
    assert "Second" in result


def test_extract_pdf_page_with_no_text_does_not_raise(dummy_app, tmp_path) -> None:
    """A page whose text extraction fails is skipped, not fatal."""
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)
    result = app.App.extract_pdf(dummy_app, str(path))
    assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# Integration: the two pieces compose the way ``App.index`` uses them
# --------------------------------------------------------------------------- #
def test_chunk_text_over_extracted_pdf_text(dummy_app, pdf_factory) -> None:
    path = pdf_factory(["alpha beta gamma delta epsilon"])
    text = app.App.extract_pdf(dummy_app, str(path))
    assert chunk(text, size=2) == ["alpha beta", "gamma delta", "epsilon"]


def test_index_like_flow_over_multiple_files(dummy_app, pdf_factory) -> None:
    """Mimic ``App.index``'s inner loop without the GUI or the model."""
    first = pdf_factory(["one two three"], name="a.pdf")
    second = pdf_factory(["four five six"], name="b.pdf")
    chunks: list[tuple[str, str]] = []
    for path in (first, second):
        text = app.App.extract_pdf(dummy_app, str(path))
        for part in app.App.chunk_text(None, text, size=2):
            chunks.append((path.name, part))
    assert chunks == [
        ("a.pdf", "one two"),
        ("a.pdf", "three"),
        ("b.pdf", "four five"),
        ("b.pdf", "six"),
    ]


# --------------------------------------------------------------------------- #
# Repository hygiene
# --------------------------------------------------------------------------- #
def test_tests_live_in_the_tests_directory(repo_root: Path) -> None:
    assert (repo_root / "tests" / "test_kb.py").is_file()
    assert (repo_root / "tests" / "conftest.py").is_file()


def test_app_module_is_loaded_from_the_repository(repo_root: Path) -> None:
    module_path = Path(sys.modules["app"].__file__).resolve()
    assert module_path.parent == repo_root
