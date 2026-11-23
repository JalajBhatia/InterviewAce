import io
import logging
from dataclasses import dataclass, field

from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


@dataclass
class ParsedResume:
    """
    Extracted text plus anything that went wrong getting it.

    Previously this returned a bare string and swallowed every failure into "",
    so a scanned image-only PDF, a corrupt file and "no resume uploaded" were all
    indistinguishable - the user was told nothing and assumed it had worked.
    Warnings are surfaced so a failed extraction is visible, and they are the
    first input to the parse-compatibility checks resume scoring will add.
    """

    text: str = ""
    warnings: list[str] = field(default_factory=list)


def parse_resume(file_content: bytes, filename: str) -> ParsedResume:
    """Extract text from a PDF or DOCX resume."""
    name = (filename or "").lower()

    if not name.endswith(SUPPORTED_EXTENSIONS):
        return ParsedResume(
            warnings=["Unsupported file format. Please upload a PDF or DOCX file."]
        )

    try:
        if name.endswith(".pdf"):
            return _parse_pdf(file_content)
        return _parse_docx(file_content)
    except Exception as exc:
        logger.warning("Resume parsing failed for %r: %s", filename, exc)
        return ParsedResume(
            warnings=["This file could not be read. It may be corrupt or password protected."]
        )


def _parse_pdf(file_content: bytes) -> ParsedResume:
    reader = PdfReader(io.BytesIO(file_content))
    pages: list[str] = []
    empty_pages = 0

    for page in reader.pages:
        # extract_text() returns None for pages with no text layer, which is what
        # a scanned page looks like. The old code did `text += page.extract_text()`
        # and raised TypeError here, straight into a bare except.
        extracted = page.extract_text() or ""
        if not extracted.strip():
            empty_pages += 1
        pages.append(extracted)

    warnings: list[str] = []
    if empty_pages:
        warnings.append(
            f"{empty_pages} of {len(reader.pages)} page(s) contained no readable text. "
            "Pages that are images rather than text are invisible to resume parsers."
        )

    return ParsedResume(text="\n".join(pages).strip(), warnings=warnings)


def _parse_docx(file_content: bytes) -> ParsedResume:
    document = Document(io.BytesIO(file_content))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]

    warnings: list[str] = []
    if document.tables:
        # Worth telling the user about: many resume parsers flatten or drop table
        # contents, so a table-based layout can lose entire sections.
        warnings.append(
            f"Document contains {len(document.tables)} table(s). Many resume parsers "
            "read tables poorly - consider a single-column layout without tables."
        )
        for table in document.tables:
            for row in table.rows:
                paragraphs.extend(cell.text for cell in row.cells)

    return ParsedResume(text="\n".join(paragraphs).strip(), warnings=warnings)
