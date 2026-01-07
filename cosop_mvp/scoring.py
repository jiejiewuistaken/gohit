from __future__ import annotations

import re
from collections.abc import Iterable

from .types import Chunk


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def score_chunk(chunk: Chunk, queries: Iterable[str]) -> int:
    txt = _normalize(chunk.text)
    score = 0
    for q in queries:
        qn = _normalize(q)
        if not qn:
            continue
        # Simple term-frequency scoring
        score += txt.count(qn) * max(1, len(qn) // 5)
    return score


def select_relevant_chunks(chunks: list[Chunk], *, queries: Iterable[str], k: int = 6) -> list[Chunk]:
    scored = [(score_chunk(c, queries), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    picked: list[Chunk] = []
    for s, c in scored:
        if s <= 0:
            continue
        picked.append(c)
        if len(picked) >= k:
            break
    return picked

