# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.2",
#   "torch>=2.1",
#   "transformers>=4.40",
#   "peft",
#   "accelerate",
#   "sacrebleu>=2.4",
#   "rouge-score>=0.1.2",
#   "nltk>=3.8",
#   "huggingface_hub",
#   "sentencepiece",
#   "protobuf<5",
# ]
# ///

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ============================================================
# ✅ SELF-CONTAINED CONFIG (no CLI args)
# ============================================================

# Which test split to run in the CSV (or embedded sample)
TEST_ID = "tran_text_en2es"

# Which language pairs to include in get_test_contexts (not required if you only run TEST_ID)
LANGUAGES = ["es"]

# Inference model
# - If your Hub repo is a full model (has config/tokenizer), set IS_PEFT_ADAPTER=False and BASE_MODEL_ID=None
# - If your Hub repo is a LoRA adapter-only repo, set IS_PEFT_ADAPTER=True and BASE_MODEL_ID=the base model id
MODEL_ID = "ifadaiml/Llama-3.1-8B-Instruct-IFAD-mt-en-es-v0.1"
IS_PEFT_ADAPTER = True
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Generation settings
DEVICE = "auto"  # "auto" | "cpu"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
USE_CHAT_TEMPLATE = True  # recommended for chat/instruct models like Llama
TRUST_REMOTE_CODE = False

# Prompt template for CausalLM (if chat_template isn't used/available)
PROMPT_TEMPLATE: Optional[str] = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output ONLY the translated text.\n\n"
    "Text:\n{text}\n\n"
    "Translation:"
)

# Data
DATA_PATH = Path("data/translation.csv")  # if missing, embedded sample will be used

# Outputs
OUT_DIR = Path("outputs") / "language_translation"

# ============================================================


def INFO(msg: str) -> None:
    print(f"[INFO] {msg}")


def ERR(msg: str) -> None:
    raise RuntimeError(msg)


CODE2LANGUAGE = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "zh": "Chinese Simplified",
}


@dataclass(frozen=True)
class _Settings:
    SYSTEM_MESSAGE: str = (
        "You are a professional translation system. "
        "Translate from {source_lang} to {target_lang}. "
        "Output ONLY the translated text, no extra words."
    )


settings = _Settings()


class CompositeTextSimilarity:
    """
    Self-contained metric wrapper.
    Returns per-example scores (0..1 for BLEU/ROUGE/METEOR where applicable).
    """

    def compute_similarity_scores(
        self, ground_truth: List[str], predictions: List[str], client: Any = None
    ) -> List[float]:
        # Placeholder for embedding similarity; keep API compatibility.
        return [0.0 for _ in range(len(ground_truth))]

    def compute_bleu_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        import sacrebleu  # type: ignore

        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            bleu = sacrebleu.sentence_bleu(pred, [ref])
            scores.append(float(bleu.score) / 100.0)
        return scores

    def compute_rouge_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            s = scorer.score(ref, pred)["rougeL"].fmeasure
            scores.append(float(s))
        return scores

    def compute_meteor_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        from nltk.translate.meteor_score import meteor_score  # type: ignore

        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            try:
                # NLTK expects tokenized lists
                scores.append(float(meteor_score([ref.split()], pred.split())))
            except LookupError:
                # If wordnet isn't available, keep runner runnable.
                scores.append(0.0)
        return scores


def _resolve_hf_token() -> str:
    # Note: HF_TOKEN is assumed to be set in your environment.
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )
    if not token:
        ERR("Missing HF token. Set environment variable HF_TOKEN.")
    return token


def _chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _maybe_strip(text: str) -> str:
    return text.strip().strip('"').strip()


class HuggingFaceHubTranslationConnector:
    """
    Hub inference connector that supports:
      - Seq2Seq models via text2text-generation
      - Causal/chat LMs via text-generation

    Fixes vs original snippet:
      - Always passes HF token to from_pretrained (gated models/adapters)
      - Supports PEFT adapter loading robustly (PeftModel.from_pretrained)
      - Uses batching (BATCH_SIZE) for throughput
      - For chat models, uses apply_chat_template when available
    """

    def __init__(
        self,
        *,
        hf_token: str,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        batch_size: int = 8,
        prompt_template: Optional[str] = None,
        use_chat_template: bool = False,
        trust_remote_code: bool = False,
        base_model_id: Optional[str] = None,
        is_peft_adapter: bool = False,
    ) -> None:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )

        self.hf_token = hf_token
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code
        self.base_model_id = base_model_id
        self.is_peft_adapter = is_peft_adapter

        # Config: try model_id; if missing (adapter-only), fall back to base_model_id.
        config = None
        try:
            config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, token=hf_token
            )
        except Exception:
            if base_model_id is None:
                raise
            config = AutoConfig.from_pretrained(
                base_model_id, trust_remote_code=trust_remote_code, token=hf_token
            )

        # Tokenizer: try model_id; if missing, fall back to base_model_id.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, trust_remote_code=trust_remote_code, token=hf_token
            )
        except Exception:
            if base_model_id is None:
                raise
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

        # Padding token for LLaMA-like tokenizers (needed for batching)
        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
            self._tokenizer, "eos_token_id", None
        ) is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))

        # Device placement strategy
        use_cuda = torch.cuda.is_available()
        if device == "cpu":
            device_id = -1
            device_map = None
        elif device == "auto":
            device_id = 0 if use_cuda else -1
            device_map = "auto" if use_cuda else None
        else:
            device_id = int(device)
            device_map = None

        torch_dtype = "auto" if use_cuda else None

        if is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=trust_remote_code,
                token=hf_token,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            task = "text2text-generation"
            self._mode = "seq2seq"
        else:
            if is_peft_adapter:
                if base_model_id is None:
                    ERR("PEFT adapter loading requires BASE_MODEL_ID.")
                from peft import PeftModel  # type: ignore

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    trust_remote_code=trust_remote_code,
                    token=hf_token,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base_model, model_id, token=hf_token)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    token=hf_token,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
            task = "text-generation"
            self._mode = "causal"

        # Pipeline:
        # - If `device_map="auto"` was used, the model is managed by accelerate and cannot be moved
        #   via the pipeline `device=` argument.
        # - Only pass `device=` when we did NOT use accelerate placement.
        pipe_kwargs: Dict[str, Any] = {"model": model, "tokenizer": self._tokenizer}
        if device_map is None:
            pipe_kwargs["device"] = device_id
        self._pipe = pipeline(task, **pipe_kwargs)

    def translate_batch(self, texts: List[str], source_lang_id: str, target_lang_id: str) -> List[str]:
        if not texts:
            return []

        if self._mode == "seq2seq":
            outs = self._pipe(
                texts,
                max_new_tokens=self.max_new_tokens,
                truncation=True,
                batch_size=self.batch_size,
            )
            preds: List[str] = []
            for o in outs:
                preds.append(_maybe_strip(str(o.get("generated_text", ""))))
            return preds

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

        outs = self._pipe(
            prompts,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
            batch_size=self.batch_size,
        )

        preds: List[str] = []
        for o in outs:
            # text-generation pipeline returns list[dict], dict has 'generated_text'
            preds.append(_maybe_strip(str(o.get("generated_text", ""))))
        return preds

    def translate(self, text: str, source_lang_id: str, target_lang_id: str) -> str:
        return self.translate_batch([text], source_lang_id, target_lang_id)[0]


class LanguageTranslationServiceConfig:
    def __init__(self, connector: HuggingFaceHubTranslationConnector) -> None:
        self.metric_assigner = CompositeTextSimilarity()
        self.metrics = {
            "rouge": self.metric_assigner.compute_rouge_score,
            "bleu": self.metric_assigner.compute_bleu_score,
            "meteor": self.metric_assigner.compute_meteor_score,
            "bge_m3_similarity": self.metric_assigner.compute_similarity_scores,
            "ada_similarity": self.metric_assigner.compute_similarity_scores,
        }
        self.connector = connector

    def get_test_contexts(self) -> List[str]:
        contexts = []
        for lan in LANGUAGES:
            if lan not in CODE2LANGUAGE:
                raise ValueError(f"Unsupported language: {lan}")
            contexts.append(f"tran_text_{lan}2en")
            contexts.append(f"tran_text_en2{lan}")
        INFO(f"Test contexts: {contexts}")
        return contexts

    def get_predictions(self, context_data: pd.DataFrame, context_id: str) -> List[str]:
        source_lang = context_id.split("_")[-1].split("2")[0]
        target_lang = context_id.split("_")[-1].split("2")[1]

        texts = [str(x) for x in context_data["question"].tolist()]
        preds: List[str] = []
        for chunk in _chunked(texts, max(1, BATCH_SIZE)):
            preds.extend(
                self.connector.translate_batch(chunk, source_lang_id=source_lang, target_lang_id=target_lang)
            )
        return preds

    def evaluate(
        self,
        context_data: pd.DataFrame,
        predictions: List[str],
        criteria: List[str] = ["rouge", "bleu", "meteor"],
    ) -> Tuple[List[str], Dict[str, List[float]], Dict[str, float]]:
        validations: Dict[str, List[float]] = defaultdict(list)
        gt = context_data["ground_truth_answer"].astype(str).tolist()

        for criterion in criteria:
            if criterion not in self.metrics:
                raise ValueError(f"Unsupported criterion: {criterion}")
            validations[criterion] = self.metrics[criterion](ground_truth=gt, predictions=predictions)

        average_scores = {criterion: mean(scores) for criterion, scores in validations.items()}
        return predictions, validations, average_scores


def _embedded_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": 1,
                "test_category": "language_translation",
                "test_id": "tran_text_en2es",
                "question": "Hello, how are you?",
                "ground_truth_answer": "Hola, ¿cómo estás?",
            },
            {
                "question_id": 2,
                "test_category": "language_translation",
                "test_id": "tran_text_en2es",
                "question": "Please translate this sentence into Spanish.",
                "ground_truth_answer": "Por favor, traduce esta oración al español.",
            },
        ]
    )


def _load_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        INFO(f"Loaded dataset from {DATA_PATH} (rows={len(df)})")
        return df
    INFO(f"{DATA_PATH} not found; using embedded sample data.")
    return _embedded_sample_df()


def _filter_context(df: pd.DataFrame, test_id: str) -> pd.DataFrame:
    required = [
        "question_id",
        "test_category",
        "test_id",
        "question",
        "ground_truth_answer",
    ]
    for c in required:
        if c not in df.columns:
            ERR(f"Missing required column '{c}' in dataset.")
    out = df[(df["test_category"] == "language_translation") & (df["test_id"] == test_id)].copy()
    out = out.sort_values(by="question_id", ascending=True).reset_index(drop=True)
    return out


def main() -> None:
    hf_token = _resolve_hf_token()

    df_all = _load_df()
    df = _filter_context(df_all, TEST_ID)
    if df.empty:
        ERR(f"No rows found for TEST_ID={TEST_ID}. Check DATA_PATH or embedded sample.")

    INFO(f"Running {TEST_ID} with model={MODEL_ID} (peft_adapter={IS_PEFT_ADAPTER})")

    connector = HuggingFaceHubTranslationConnector(
        hf_token=hf_token,
        model_id=MODEL_ID,
        device=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        prompt_template=PROMPT_TEMPLATE,
        use_chat_template=USE_CHAT_TEMPLATE,
        trust_remote_code=TRUST_REMOTE_CODE,
        base_model_id=BASE_MODEL_ID,
        is_peft_adapter=IS_PEFT_ADAPTER,
    )
    config = LanguageTranslationServiceConfig(connector=connector)

    preds = config.get_predictions(context_data=df, context_id=TEST_ID)
    _, per_example, averages = config.evaluate(context_data=df, predictions=preds)

    INFO(f"Average scores: {averages}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{TEST_ID}__predictions.csv"
    out_df = df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(out_path, index=False)
    INFO(f"Saved predictions to {out_path}")

    # Also save a metrics summary for convenience.
    metrics_path = OUT_DIR / f"{TEST_ID}__metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"model_id: {MODEL_ID}\n")
        f.write(f"test_id: {TEST_ID}\n")
        f.write(f"avg_scores: {averages}\n")
        f.write(f"per_example_keys: {list(per_example.keys())}\n")
    INFO(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

