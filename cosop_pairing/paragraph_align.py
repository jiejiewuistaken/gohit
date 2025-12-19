from __future__ import annotations

from dataclasses import dataclass

from .docx_extract import Paragraph


@dataclass(frozen=True)
class ParagraphPair:
    en: Paragraph
    es: Paragraph


def align_paragraphs_monotonic_by_length(
    en_paragraphs: list[Paragraph],
    es_paragraphs: list[Paragraph],
    *,
    window: int = 5,
) -> list[ParagraphPair]:
    """
    Lightweight monotonic paragraph aligner (no LLM).

    For each EN paragraph in order, chooses the "closest length" ES paragraph from a
    small forward window starting at the current ES pointer.
    """
    pairs: list[ParagraphPair] = []
    j = 0
    for en in en_paragraphs:
        if j >= len(es_paragraphs):
            break

        # Search within the next `window` ES paragraphs for closest char length.
        candidates = []
        for k in range(j, min(len(es_paragraphs), j + window)):
            es = es_paragraphs[k]
            candidates.append((abs(len(en.text) - len(es.text)), k, es))
        candidates.sort(key=lambda t: (t[0], t[1]))
        _, best_k, best_es = candidates[0]

        pairs.append(ParagraphPair(en=en, es=best_es))
        j = best_k + 1

    return pairs

