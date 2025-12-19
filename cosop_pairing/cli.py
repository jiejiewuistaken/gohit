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
        help="Model (OpenAI) or deployment name (Azure OpenAI) for structured pairing.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL (e.g. private endpoint). Also supports env OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint URL. Also supports env AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--azure-api-version",
        type=str,
        default=None,
        help="Azure OpenAI API version. Also supports env AZURE_OPENAI_API_VERSION.",
    )
    parser.add_argument(
        "--use-azure",
        action="store_true",
        help="Force Azure OpenAI mode (otherwise auto-detected from env AZURE_OPENAI_*).",
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
        base_url=args.base_url,
        use_azure=True if args.use_azure else None,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        paragraph_window=args.paragraph_window,
        max_sentences_per_call=args.max_sentences_per_call,
    )
    written = run_pipeline(cfg)
    print(f"Wrote {len(written)} files to {cfg.output_dir}")


if __name__ == "__main__":
    main()

