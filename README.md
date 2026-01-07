# COSOP MVP (local-files → COSOP draft)

This is a small MVP CLI that reads local reference files (PDF/DOCX/TXT/MD), extracts text, and generates a **template-aligned COSOP draft** (Markdown; optional DOCX).

## Setup

```bash
pip3 install -r requirements.txt
```

## Generate a COSOP draft

```bash
python3 -m cosop_mvp \
  --country "Exampleland" \
  --period-start 2026 \
  --period-end 2030 \
  --input "path/to/reference1.pdf" \
  --input "path/to/reference2.docx" \
  --out-dir out
```

Outputs:
- `out/cosop.md`
- `out/cosop.docx` (if `--docx` is set)

## Notes (MVP limitations)
- This MVP **does not use an LLM**. It uses keyword-based retrieval from your input documents to populate each COSOP section with relevant excerpts + simple citations.
- Tables (Table 1–3 and RMF) are created as **fillable scaffolds** and will show `TBD` where structured data is missing.

