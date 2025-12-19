from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .discover import DocPair, discover_doc_pairs
from .docx_extract import extract_paragraphs
from .llm_pair import pair_sentences_structured
from .models import Sentence, SentencePair, SentencePairs
from .paragraph_align import align_paragraphs_monotonic_by_length
from .sentence_split import split_into_sentences


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    output_dir: Path
    model: str = "gpt-4o-mini"
    # For private OpenAI-compatible endpoints (settable via OPENAI_BASE_URL).
    base_url: str | None = None
    # For Azure OpenAI (settable via AZURE_OPENAI_* env vars).
    use_azure: bool | None = None
    azure_endpoint: str | None = None
    azure_api_version: str | None = None
    paragraph_window: int = 5
    max_sentences_per_call: int = 40


def _sentences_from_paragraph(
    *,
    paragraph_text: str,
    document_id: str,
    page_number: int,
    start_index: int,
) -> list[Sentence]:
    sents = split_into_sentences(paragraph_text)
    out: list[Sentence] = []
    for i, s in enumerate(sents):
        out.append(
            Sentence(
                content=s,
                document_id=document_id,
                page_number=page_number,
                sentence_index=start_index + i,
            )
        )
    return out


def _write_sentence_pairs(path: Path, pairs: SentencePairs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_doc_pair(pair: DocPair, *, cfg: PipelineConfig) -> SentencePairs:
    en_paragraphs = extract_paragraphs(pair.en_docx)
    es_paragraphs = extract_paragraphs(pair.es_docx)

    paragraph_pairs = align_paragraphs_monotonic_by_length(
        en_paragraphs,
        es_paragraphs,
        window=cfg.paragraph_window,
    )

    all_pairs: list[SentencePair] = []
    en_sentence_idx = 0
    es_sentence_idx = 0

    for pp in paragraph_pairs:
        # .docx does not reliably provide page numbers; default to 0.
        en_sents = _sentences_from_paragraph(
            paragraph_text=pp.en.text,
            document_id=pair.document_id,
            page_number=0,
            start_index=en_sentence_idx,
        )
        es_sents = _sentences_from_paragraph(
            paragraph_text=pp.es.text,
            document_id=pair.document_id,
            page_number=0,
            start_index=es_sentence_idx,
        )

        en_sentence_idx += len(en_sents)
        es_sentence_idx += len(es_sents)

        if not en_sents or not es_sents:
            continue

        # Keep requests small and predictable.
        en_sents = en_sents[: cfg.max_sentences_per_call]
        es_sents = es_sents[: cfg.max_sentences_per_call]

        aligned = pair_sentences_structured(
            en_sentences=en_sents,
            es_sentences=es_sents,
            model=cfg.model,
            base_url=cfg.base_url,
            use_azure=cfg.use_azure,
            azure_endpoint=cfg.azure_endpoint,
            azure_api_version=cfg.azure_api_version,
        )
        all_pairs.extend(aligned.sentence_pairs)

    return SentencePairs(sentence_pairs=all_pairs)


def run_pipeline(cfg: PipelineConfig) -> list[Path]:
    doc_pairs = discover_doc_pairs(cfg.root)
    written: list[Path] = []

    for p in doc_pairs:
        pairs = process_doc_pair(p, cfg=cfg)
        out_path = cfg.output_dir / f"{p.document_id}.sentence_pairs.json"
        _write_sentence_pairs(out_path, pairs)
        written.append(out_path)

    return written

