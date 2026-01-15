import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from huggingface_hub import login


# ----------------------------
# Config (edit in Space Variables/Secrets)
# ----------------------------

# Base model (for gated models like Llama, Space needs HF_TOKEN in Secrets)
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

# LoRA adapter-only repo (your fine-tuned adapter)
ADAPTER_REPO_ID = os.environ.get(
    "ADAPTER_REPO_ID", "ifadaiml/Llama-3.1-8B-Instruct-IFAD-mt-en-es-v0.2"
)

# In Space Secrets if base model is gated / adapter repo is private
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)

# Match the evaluation script behavior (prompt + chat template usage)
USE_CHAT_TEMPLATE = True
TRUST_REMOTE_CODE = False

CODE2LANGUAGE = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "zh": "Chinese Simplified",
}

LANG_CODE_CHOICES = list(CODE2LANGUAGE.keys())

PROMPT_TEMPLATE: Optional[str] = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output ONLY the translated text.\n\n"
    "Text:\n{text}\n\n"
    "Translation:"
)


def _maybe_strip(text: str) -> str:
    return text.strip().strip('"').strip()


def _first_generated_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, list) and x:
        return _first_generated_dict(x[0])
    return {}


class HuggingFaceHubTranslationConnector:
    """
    Same idea as your testPlan connector, but:
    - Loads adapter ONCE (base weights only once)
    - Produces both base and finetuned outputs via `adapter_enabled` switch
      using `with model.disable_adapter(): ...` around the pipeline call.
    """

    def __init__(
        self,
        *,
        hf_token: str,
        base_model_id: str,
        adapter_model_id: str,
        prompt_template: Optional[str] = None,
        use_chat_template: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
        )
        from peft import PeftModel  # type: ignore

        self.hf_token = hf_token
        self.base_model_id = base_model_id
        self.adapter_model_id = adapter_model_id
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code

        use_cuda = torch.cuda.is_available()
        device_map = "auto" if use_cuda else None
        torch_dtype = "auto" if use_cuda else None

        # Tokenizer: try adapter first (if it contains tokenizer), else base.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                adapter_model_id,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

        # Decoder-only models: left padding for better batching correctness
        try:
            self._tokenizer.padding_side = "left"
        except Exception:
            pass

        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
            self._tokenizer, "eos_token_id", None
        ) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self._model = PeftModel.from_pretrained(base, adapter_model_id, token=hf_token)
        self._model.eval()

        pipe_kwargs: Dict[str, Any] = {"model": self._model, "tokenizer": self._tokenizer}
        self._pipe = pipeline("text-generation", **pipe_kwargs)

    def _build_prompts(self, texts: List[str], source_lang_id: str, target_lang_id: str) -> List[str]:
        source_lang = CODE2LANGUAGE.get(source_lang_id, source_lang_id)
        target_lang = CODE2LANGUAGE.get(target_lang_id, target_lang_id)
        prompt_template = self.prompt_template or (
            "Translate the following text from {source_lang} to {target_lang}. "
            "Return only the translation.\n\nText:\n{text}\n\nTranslation:"
        )

        prompts: List[str] = []
        if (
            self.use_chat_template
            and hasattr(self._tokenizer, "apply_chat_template")
            and getattr(self._tokenizer, "chat_template", None)
        ):
            for t in texts:
                messages = [
                    {"role": "system", "content": "You are a professional translation system."},
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            source_lang=source_lang, target_lang=target_lang, text=t
                        ),
                    },
                ]
                prompts.append(
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
        else:
            for t in texts:
                prompts.append(
                    prompt_template.format(source_lang=source_lang, target_lang=target_lang, text=t)
                )
        return prompts

    @torch.inference_mode()
    def translate_one(
        self,
        *,
        text: str,
        source_lang_id: str,
        target_lang_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        adapter_enabled: bool,
    ) -> str:
        prompts = self._build_prompts([text], source_lang_id, target_lang_id)
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=float(temperature) > 0,
            temperature=float(temperature),
            top_p=float(top_p),
            return_full_text=False,
            batch_size=1,
        )

        if adapter_enabled:
            outs = self._pipe(prompts, **gen_kwargs)
        else:
            # Base output: run the same pipeline, but with adapter disabled.
            with self._model.disable_adapter():
                outs = self._pipe(prompts, **gen_kwargs)

        d = _first_generated_dict(outs)
        return _maybe_strip(str(d.get("generated_text", "")))


_CONNECTOR: Optional[HuggingFaceHubTranslationConnector] = None


def get_connector() -> HuggingFaceHubTranslationConnector:
    global _CONNECTOR
    if _CONNECTOR is None:
        if HF_TOKEN:
            login(token=HF_TOKEN, add_to_git_credential=False)
        _CONNECTOR = HuggingFaceHubTranslationConnector(
            hf_token=HF_TOKEN or "",
            base_model_id=BASE_MODEL_ID,
            adapter_model_id=ADAPTER_REPO_ID,
            prompt_template=PROMPT_TEMPLATE,
            use_chat_template=USE_CHAT_TEMPLATE,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
    return _CONNECTOR


def generate_translation(
    src_code: str,
    tgt_code: str,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    if not text or not text.strip():
        return "", ""

    c = get_connector()
    base_out = c.translate_one(
        text=text,
        source_lang_id=src_code,
        target_lang_id=tgt_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        adapter_enabled=False,
    )
    ft_out = c.translate_one(
        text=text,
        source_lang_id=src_code,
        target_lang_id=tgt_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        adapter_enabled=True,
    )
    return base_out, ft_out


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
            src_lang = gr.Dropdown(choices=LANG_CODE_CHOICES, value="en", label="From (lang code)")
            tgt_lang = gr.Dropdown(choices=LANG_CODE_CHOICES, value="es", label="To (lang code)")
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

