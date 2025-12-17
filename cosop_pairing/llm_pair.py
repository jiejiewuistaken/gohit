from __future__ import annotations

import json
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


def _get_client(
    *,
    base_url: str | None = None,
    use_azure: bool | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
) -> Any:
    load_dotenv()

    # --- Azure OpenAI ---
    auto_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_KEY"))
    if use_azure is None:
        use_azure = auto_azure

    if use_azure:
        from openai import AzureOpenAI

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"

        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set (env var or CLI).")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set (env var).")

        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    # --- Public/Private OpenAI-compatible endpoint ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (env var or .env).")

    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def pair_sentences_structured(
    *,
    en_sentences: list[Sentence],
    es_sentences: list[Sentence],
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    use_azure: bool | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
) -> SentencePairs:
    """
    Use OpenAI Structured Outputs to align two lists of sentences.
    """
    client = _get_client(
        base_url=base_url,
        use_azure=use_azure,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
    )

    payload = {
        "english_sentences": [s.model_dump() for s in en_sentences],
        "spanish_sentences": [s.model_dump() for s in es_sentences],
    }
    user_content = (
        "Align these sentences into translation-equivalent pairs.\n"
        "Return ONLY the structured output.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    # Prefer the parse helper when available (Structured Outputs).
    # The OpenAI Python SDK has evolved; we support the common variants.
    if hasattr(client, "responses") and hasattr(client.responses, "parse"):
        resp: Any = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_content,
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

    if hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(client.chat.completions, "parse"):
        comp: Any = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=SentencePairs,
        )
        return comp.choices[0].message.parsed

    if hasattr(client, "beta") and hasattr(client.beta, "chat") and hasattr(client.beta.chat.completions, "parse"):
        comp: Any = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format=SentencePairs,
        )
        return comp.choices[0].message.parsed

    raise RuntimeError("This openai SDK version does not support structured parsing (parse).")

