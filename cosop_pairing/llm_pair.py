from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .models import Sentence, SentencePairs


SYSTEM_PROMPT = """You align English and Spanish sentences for translation fine-tuning.

Rules:
- You will be given two lists of Sentence objects (sentence_1 list is English, sentence_2 list is Spanish).
- Return SentencePairs where each pair is a translation-equivalent mapping.
- Use the provided sentences exactly; do not invent or rewrite content.
- Prefer 1-to-1 pairing. If needed, you may skip sentences that are headers, bullets, table fragments, or non-translatable noise.
- Do not create pairs with empty content.
"""


def _get_client() -> OpenAI:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (env var or .env).")
    return OpenAI()


def pair_sentences_structured(
    *,
    en_sentences: list[Sentence],
    es_sentences: list[Sentence],
    model: str = "gpt-4o-mini",
) -> SentencePairs:
    """
    Use OpenAI Structured Outputs to align two lists of sentences.
    """
    client = _get_client()

    payload = {
        "english_sentences": [s.model_dump() for s in en_sentences],
        "spanish_sentences": [s.model_dump() for s in es_sentences],
    }

    # Prefer the parse helper when available (Structured Outputs).
    # The OpenAI Python SDK has evolved; we support the common variants.
    if hasattr(client, "responses") and hasattr(client.responses, "parse"):
        resp: Any = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Align these sentences into translation-equivalent pairs.\n\n"
                    + str(payload),
                },
            ],
            text_format=SentencePairs,
        )
        # responses.parse returns an object with `output_parsed` in many versions
        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            # Fallback: some versions expose `output[0].content[0].parsed`
            parsed = resp.output[0].content[0].parsed  # type: ignore[attr-defined]
        return parsed

    if hasattr(client, "beta") and hasattr(client.beta, "chat") and hasattr(client.beta.chat.completions, "parse"):
        comp: Any = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Align these sentences into translation-equivalent pairs.\n\n"
                    + str(payload),
                },
            ],
            response_format=SentencePairs,
        )
        return comp.choices[0].message.parsed

    raise RuntimeError("This openai SDK version does not support structured parsing (parse).")

