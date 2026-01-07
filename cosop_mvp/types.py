from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Citation:
    path: Path
    page: int | None  # 1-based page for PDFs; otherwise None

    def label(self) -> str:
        if self.page is None:
            return f"{self.path.name}"
        return f"{self.path.name} p.{self.page}"


@dataclass(frozen=True)
class Chunk:
    text: str
    citation: Citation


@dataclass(frozen=True)
class ExtractedDocument:
    path: Path
    kind: str  # pdf/docx/txt/md/unknown
    chunks: list[Chunk]

