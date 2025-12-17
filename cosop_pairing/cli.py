from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="COSOP EN/ES sentence pairing pipeline.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing COSOP/EN English and COSOP/ES Spanish",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sentence_pairs"),
        help="Where to write SentencePairs JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for structured pairing.",
    )
    parser.add_argument(
        "--paragraph-window",
        type=int,
        default=5,
        help="How far to look ahead when aligning paragraphs by length.",
    )
    parser.add_argument(
        "--max-sentences-per-call",
        type=int,
        default=40,
        help="Safety limit per LLM call (per paragraph pair).",
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        root=args.root,
        output_dir=args.output_dir,
        model=args.model,
        paragraph_window=args.paragraph_window,
        max_sentences_per_call=args.max_sentences_per_call,
    )
    written = run_pipeline(cfg)
    print(f"Wrote {len(written)} files to {cfg.output_dir}")


if __name__ == "__main__":
    main()

