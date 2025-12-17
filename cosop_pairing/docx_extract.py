from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import docx


@dataclass(frozen=True)
class Paragraph:
    text: str
    paragraph_index: int


def extract_paragraphs(docx_path: Path) -> list[Paragraph]:
    """
    Extract non-empty paragraphs from a .docx file.

    Note: .docx does not reliably encode page numbers; we treat page_number as 0
    downstream unless you add a separate pagination step.
    """
    d = docx.Document(str(docx_path))
    out: list[Paragraph] = []
    for i, p in enumerate(d.paragraphs):
        text = (p.text or "").strip()
        if not text:
            continue
        out.append(Paragraph(text=text, paragraph_index=i))
    return out

