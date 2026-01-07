from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .ingest import ingest_paths
from .render import CosopMeta, render_docx, render_markdown
from .scoring import select_relevant_chunks
from .template import cosop_sections


def _build_sections(*, all_chunks, console: Console) -> dict[str, str]:
    sections_out: dict[str, str] = {}
    specs = cosop_sections()
    for spec in specs:
        picked = select_relevant_chunks(all_chunks, queries=spec.queries, k=6)
        if not picked:
            sections_out[spec.key] = ""
            continue
        bullets = []
        for ch in picked:
            txt = ch.text.strip().replace("\n", " ")
            # Keep excerpts short to avoid dumping whole documents.
            if len(txt) > 380:
                txt = txt[:380].rstrip() + "…"
            bullets.append(f"- {txt} ({ch.citation.label()})")
        sections_out[spec.key] = "\n".join(bullets)
    return sections_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosop_mvp", description="Local files → COSOP template draft (MVP).")
    p.add_argument("--input", action="append", required=True, help="Input file path (PDF/DOCX/TXT/MD). Can be repeated.")
    p.add_argument("--out-dir", default="out", help="Output directory (default: out).")
    p.add_argument("--country", required=True, help="Country name.")
    p.add_argument("--period-start", required=True, help="COSOP period start (e.g., 2026).")
    p.add_argument("--period-end", required=True, help="COSOP period end (e.g., 2030).")
    p.add_argument("--docx", action="store_true", help="Also generate out/cosop.docx (simple conversion).")
    p.add_argument("--eb-session", default=None)
    p.add_argument("--eb-meeting-date", default=None)
    p.add_argument("--document-code", default=None)
    p.add_argument("--agenda-item", default=None)
    p.add_argument("--sec-date", default=None)
    p.add_argument("--original-language", default=None)
    p.add_argument("--useful-references", default=None, help="Comma-separated references for front matter.")

    args = p.parse_args(argv)
    console = Console()

    in_paths = [Path(x).expanduser().resolve() for x in args.input]
    for ip in in_paths:
        if not ip.exists():
            console.print(f"[red]Missing input file:[/red] {ip}")
            return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Ingesting[/bold] {len(in_paths)} files…")
    docs = ingest_paths(in_paths)
    all_chunks = [c for d in docs for c in d.chunks if c.text.strip()]

    console.print(f"[bold]Extracted[/bold] {len(all_chunks)} text chunks.")
    sections = _build_sections(all_chunks=all_chunks, console=console)

    meta = CosopMeta(
        country=args.country,
        period_start=args.period_start,
        period_end=args.period_end,
        eb_session=args.eb_session,
        eb_meeting_date=args.eb_meeting_date,
        document_code=args.document_code,
        agenda_item=args.agenda_item,
        sec_date=args.sec_date,
        original_language=args.original_language,
        useful_references=args.useful_references,
    )

    evidence_index = [str(p.name) for p in in_paths]
    md = render_markdown(meta, sections=sections, evidence_index=evidence_index)

    md_path = out_dir / "cosop.md"
    md_path.write_text(md, encoding="utf-8")
    console.print(f"[green]Wrote[/green] {md_path}")

    if args.docx:
        docx_path = out_dir / "cosop.docx"
        render_docx(docx_path, md)
        console.print(f"[green]Wrote[/green] {docx_path}")

    console.print("[bold green]Done.[/bold green]")
    return 0

