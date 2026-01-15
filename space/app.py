import os
from typing import Tuple

import gradio as gr
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Config (edit in Space Variables/Secrets)
# ----------------------------

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
# Your LoRA adapter repo (contains adapter weights + tokenizer if you pushed it)
ADAPTER_REPO_ID = os.environ.get(
    "ADAPTER_REPO_ID", "ifadaiml/Llama-3.1-8B-Instruct-IFAD-mt-en-es-v0.2"
)

# If BASE_MODEL_ID is gated (e.g. Llama), set this in Space Secrets
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)


LANGS = {
    "English": "English",
    "Spanish": "Spanish",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "French": "French",
    "German": "German",
    "Korean": "Korean",
}


def build_prompt(src_lang: str, tgt_lang: str, text: str) -> str:
    src = LANGS.get(src_lang, src_lang)
    tgt = LANGS.get(tgt_lang, tgt_lang)
    return f"Translate {src} to {tgt}.\n\n{src}: {text.strip()}\n{tgt}:"


def _pick_dtype() -> torch.dtype:
    # Prefer bf16 on modern GPUs; fallback to fp16; CPU uses float32.
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere+
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_models() -> tuple[PeftModel, AutoTokenizer]:
    if HF_TOKEN:
        # Works for both gated base models and private adapter repos.
        login(token=HF_TOKEN, add_to_git_credential=False)

    dtype = _pick_dtype()
    device_map = "auto" if torch.cuda.is_available() else None

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    # Important for generation speed/memory in training, but harmless here.
    if getattr(base.config, "use_cache", None) is True:
        base.config.use_cache = True

    # Tokenizer: prefer adapter repo if it contains tokenizer; else fall back to base.
    try:
        tok = AutoTokenizer.from_pretrained(ADAPTER_REPO_ID, token=HF_TOKEN, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ft = PeftModel.from_pretrained(base, ADAPTER_REPO_ID, token=HF_TOKEN)
    ft.eval()
    return ft, tok


_FT_MODEL = None
_TOKENIZER = None


def get_models():
    global _FT_MODEL, _TOKENIZER
    if _FT_MODEL is None or _TOKENIZER is None:
        _FT_MODEL, _TOKENIZER = load_models()
    return _FT_MODEL, _TOKENIZER


@torch.inference_mode()
def generate_translation(
    src_lang: str,
    tgt_lang: str,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    if not text or not text.strip():
        return "", ""

    model, tok = get_models()
    prompt = build_prompt(src_lang, tgt_lang, text)

    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=float(temperature) > 0,
        temperature=float(temperature),
        top_p=float(top_p),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # Fine-tuned output (with adapter)
    out_ft = model.generate(**inputs, **gen_kwargs)
    ft_text = tok.decode(out_ft[0], skip_special_tokens=True)

    # Base output (same underlying base model, but adapter disabled)
    try:
        with model.disable_adapter():
            out_base = model.generate(**inputs, **gen_kwargs)
    except Exception:
        # Fallback if peft version doesn't support disable_adapter()
        out_base = model.base_model.generate(**inputs, **gen_kwargs)
    base_text = tok.decode(out_base[0], skip_special_tokens=True)

    # Return only the completion after the prompt (best-effort)
    def strip_prompt(full: str) -> str:
        return full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()

    return strip_prompt(base_text), strip_prompt(ft_text)


def swap_langs(src: str, tgt: str) -> tuple[str, str]:
    return tgt, src


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Translation (Base vs Fine-tuned)") as demo:
        gr.Markdown(
            "### Translation Demo\n"
            "Compare **base model** vs **fine-tuned (LoRA adapter)** outputs.\n\n"
            f"- Base: `{BASE_MODEL_ID}`\n"
            f"- Adapter: `{ADAPTER_REPO_ID}`"
        )

        with gr.Row():
            src_lang = gr.Dropdown(choices=list(LANGS.keys()), value="English", label="From")
            tgt_lang = gr.Dropdown(choices=list(LANGS.keys()), value="Spanish", label="To")
            switch = gr.Button("Switch")

        switch.click(fn=swap_langs, inputs=[src_lang, tgt_lang], outputs=[src_lang, tgt_lang])

        inp = gr.Textbox(lines=4, label="Input text", placeholder="Type a sentence to translate...")

        with gr.Accordion("Generation settings", open=False):
            max_new_tokens = gr.Slider(16, 512, value=128, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")

        btn = gr.Button("Translate")

        with gr.Row():
            out_base = gr.Textbox(lines=6, label="Base model output")
            out_ft = gr.Textbox(lines=6, label="Fine-tuned model output (LoRA)")

        btn.click(
            fn=generate_translation,
            inputs=[src_lang, tgt_lang, inp, max_new_tokens, temperature, top_p],
            outputs=[out_base, out_ft],
        )

        gr.Markdown(
            "Notes:\n"
            "- If the base model is gated (e.g. Llama), set Space **Secrets** `HF_TOKEN`.\n"
            "- For 8B models you typically need a **GPU Space**."
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    # Spaces uses `app.py` at repo root by default.
    # If you copy this file to Space root, it will work as-is.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

