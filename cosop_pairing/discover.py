from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocPair:
    document_id: str
    en_docx: Path
    es_docx: Path


def _first_docx_in_dir(d: Path) -> Path | None:
    if not d.exists() or not d.is_dir():
        return None
    docx_files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".docx"])
    return docx_files[0] if docx_files else None


def discover_doc_pairs(root: Path) -> list[DocPair]:
    """
    Discover EN/ES docx pairs under:

      COSOP/EN English/EBxxxxxx/
      COSOP/ES Spanish/EBxxxxxx/

    Assumptions:
    - document_id is the EB folder name (e.g., EB123456)
    - each EB folder contains at least one .docx; if multiple, the first (sorted) is used.
    """
    en_base = root / "COSOP" / "EN English"
    es_base = root / "COSOP" / "ES Spanish"

    if not en_base.exists() or not es_base.exists():
        return []

    pairs: list[DocPair] = []

    for en_folder in sorted([p for p in en_base.iterdir() if p.is_dir() and p.name.upper().startswith("EB")]):
        document_id = en_folder.name
        es_folder = es_base / document_id
        en_docx = _first_docx_in_dir(en_folder)
        es_docx = _first_docx_in_dir(es_folder)
        if not en_docx or not es_docx:
            continue
        pairs.append(DocPair(document_id=document_id, en_docx=en_docx, es_docx=es_docx))

    return pairs

