from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")

# A deliberately simple sentence splitter that works reasonably for COSOP-style prose.
# If you need higher quality, swap this with a proper segmenter later.
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'“”‘’\(\[])?[A-ZÁÉÍÓÚÑÜ¿¡])")


def normalize_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []

    # If there are no clear sentence boundaries, keep as one sentence.
    if not any(p in text for p in (".", "!", "?")):
        return [text]

    parts = _SENT_END_RE.split(text)
    out = [p.strip() for p in parts if p.strip()]
    return out or [text]

