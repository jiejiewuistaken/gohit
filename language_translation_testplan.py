from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------
# Logging helpers (keep style)
# ----------------------------


def INFO(msg: str) -> None:
    print(f"[INFO] {msg}")


def ERR(msg: str) -> None:
    raise RuntimeError(msg)


# ----------------------------
# Dicts for language conversions (kept from your file)
# ----------------------------

CODE2LANGUAGE = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "zh": "Chinese Simplified",
}
LANGUAGE2CODE = {v: k for k, v in CODE2LANGUAGE.items()}


# ----------------------------
# Minimal "settings" replacement
# ----------------------------


@dataclass(frozen=True)
class _Settings:
    # Kept to mirror your usage: SYSTEM_MESSAGE.format(source_lang=..., target_lang=...)
    SYSTEM_MESSAGE: str = (
        "You are a professional translation system. "
        "Translate from {source_lang} to {target_lang}. "
        "Output ONLY the translated text, no extra words."
    )


settings = _Settings()


# ----------------------------
# Minimal schema/exporter replacement (self-contained)
# ----------------------------


class QuestionsSchema:
    """
    Placeholder for your project's DB schema class.
    In this self-contained runner we read from CSV, but we keep the name.
    """


class LocalCSVExporter:
    """
    Drop-in replacement for get_exporter(args).read_table(...).
    Expects a CSV with at least:
      - question_id (int)
      - test_category (str) == "language_translation"
      - test_id (str) e.g. "tran_text_en2es"
      - question (str) source text
      - ground_truth_answer (str) reference translation
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def read_table(self, _schema: Any, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        if filters:
            for k, v in filters.items():
                if k not in df.columns:
                    ERR(f"Missing required column '{k}' in {self.data_path}")
                df = df[df[k] == v]
        return df


def get_exporter(args: argparse.Namespace) -> LocalCSVExporter:
    return LocalCSVExporter(Path(args.data_path))


# ----------------------------
# Minimal metric implementation (self-contained)
# ----------------------------


class CompositeTextSimilarity:
    """
    Keep the class name and method names used by your code.
    In this self-contained version:
      - compute_bleu_score uses sacrebleu (sentence-level, averaged)
      - compute_rouge_score uses rouge-score (rougeL F1, averaged)
      - compute_meteor_score uses nltk (meteor_score, averaged)
    """

    def compute_similarity_scores(self, ground_truth: List[str], predictions: List[str], client: Any = None) -> List[float]:
        # You asked for HF translation inference; embedding similarity clients are project-specific.
        # Keep API but return a placeholder score so the pipeline remains runnable if requested.
        # For real embedding similarity, plug your own client here.
        return [0.0 for _ in range(len(ground_truth))]

    def compute_bleu_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        try:
            import sacrebleu  # type: ignore
        except Exception as e:
            ERR(f"BLEU metric requires 'sacrebleu'. Install requirements.txt. Original error: {e}")

        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            bleu = sacrebleu.sentence_bleu(pred, [ref])
            scores.append(float(bleu.score) / 100.0)  # normalize to 0..1
        return scores

    def compute_rouge_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        try:
            from rouge_score import rouge_scorer  # type: ignore
        except Exception as e:
            ERR(f"ROUGE metric requires 'rouge-score'. Install requirements.txt. Original error: {e}")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            s = scorer.score(ref, pred)["rougeL"].fmeasure
            scores.append(float(s))
        return scores

    def compute_meteor_score(self, ground_truth: List[str], predictions: List[str]) -> List[float]:
        try:
            from nltk.translate.meteor_score import meteor_score  # type: ignore
        except Exception as e:
            ERR(f"METEOR metric requires 'nltk'. Install requirements.txt. Original error: {e}")

        scores: List[float] = []
        for ref, pred in zip(ground_truth, predictions):
            # meteor_score expects tokenized inputs; NLTK may require extra corpora (wordnet).
            try:
                scores.append(float(meteor_score([ref.split()], pred.split())))
            except LookupError:
                # Keep runner self-contained: if wordnet isn't available, fall back to 0.0.
                scores.append(0.0)
        return scores


# ----------------------------
# HuggingFace Hub translation connector (your requested inference)
# ----------------------------


class HuggingFaceHubTranslationConnector:
    """
    HF Hub inference connector that supports:
      - Seq2Seq translation models (AutoModelForSeq2SeqLM + text2text-generation)
      - Causal LMs / chat LMs (e.g. LLaMA) (AutoModelForCausalLM + text-generation)
    For CausalLM, we construct a translation prompt (optionally using chat_template).
    """

    def __init__(
        self,
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
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:
            ERR(f"HuggingFace inference requires 'transformers' (+ torch). Install requirements.txt. Original error: {e}")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code
        self.base_model_id = base_model_id
        self.is_peft_adapter = is_peft_adapter

        # Some repos (often PEFT adapters) do not ship a usable config/tokenizer.
        # We support falling back to a base model's config/tokenizer when provided.
        config = None
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except ValueError:
            if base_model_id is None:
                raise
        except OSError:
            if base_model_id is None:
                raise

        # Tokenizer: try model_id first; if missing, fall back to base_model_id.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
        except Exception:
            if base_model_id is None:
                raise
            self._tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=trust_remote_code)

        if config is None and base_model_id is not None:
            config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)

        # Decide model family (Seq2Seq vs CausalLM)
        is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
        if is_seq2seq:
            # If weights repo is missing config, config comes from base_model_id above.
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=trust_remote_code,
            )
            task = "text2text-generation"
            self._mode = "seq2seq"
        else:
            if is_peft_adapter:
                if base_model_id is None:
                    ERR("PEFT adapter loading requires --base_model_id")
                try:
                    from peft import PeftModel  # type: ignore
                except Exception as e:
                    ERR(f"PEFT adapter loading requires 'peft'. Install requirements.txt. Original error: {e}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    trust_remote_code=trust_remote_code,
                )
                model = PeftModel.from_pretrained(base_model, model_id)
            else:
                # If weights repo is missing config, config comes from base_model_id above.
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=trust_remote_code,
                )
            task = "text-generation"
            self._mode = "causal"

        if device == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
        else:
            device_id = int(device)

        # Pipeline for either mode.
        self._pipe = pipeline(task, model=model, tokenizer=self._tokenizer, device=device_id)

    def translate(self, text: str, source_lang_id: str, target_lang_id: str) -> str:
        # Keep signature used by your testplan.
        if self._mode == "seq2seq":
            out = self._pipe(text, max_new_tokens=self.max_new_tokens, truncation=True)
            if not out:
                return ""
            return str(out[0].get("generated_text", "")).strip()

        # CausalLM path (e.g. LLaMA): build a prompt.
        source_lang = CODE2LANGUAGE.get(source_lang_id, source_lang_id)
        target_lang = CODE2LANGUAGE.get(target_lang_id, target_lang_id)
        prompt_template = self.prompt_template or (
            "Translate the following text from {source_lang} to {target_lang}. "
            "Return only the translation.\n\n"
            "Text:\n{text}\n\n"
            "Translation:"
        )

        if self.use_chat_template and hasattr(self._tokenizer, "apply_chat_template") and getattr(self._tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "You are a professional translation system."},
                {
                    "role": "user",
                    "content": prompt_template.format(source_lang=source_lang, target_lang=target_lang, text=text),
                },
            ]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out = self._pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )
        else:
            prompt = prompt_template.format(source_lang=source_lang, target_lang=target_lang, text=text)
            out = self._pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )

        if not out:
            return ""
        generated = str(out[0].get("generated_text", "")).strip()
        return generated


# ----------------------------
# Minimal base config classes (keep names)
# ----------------------------


class BaseLLMTestConfiguration:
    pass


class BaseSpecialTestConfiguration:
    pass


# ----------------------------
# Your classes (kept, with small fixes to run)
# ----------------------------


class LanguageTranslationLLMConfig(BaseLLMTestConfiguration):
    """Test configuration for language translation (LLM prompts)."""

    def __init__(self, args):
        self.args = args
        self.settings = settings

        self.metric_assigner = CompositeTextSimilarity()
        self.metrics = {
            "rouge": self.metric_assigner.compute_rouge_score,
            "bleu": self.metric_assigner.compute_bleu_score,
            "meteor": self.metric_assigner.compute_meteor_score,
            # kept keys from your original file (placeholders in self-contained runner)
            "bge_m3_similarity": self.metric_assigner.compute_similarity_scores,
            "ada_similarity": self.metric_assigner.compute_similarity_scores,
        }
        self.exporter = get_exporter(args)

    def get_data(self) -> Path:
        # not needed in this self-contained runner; kept for interface compatibility
        return Path(self.args.data_path)

    def get_output_path(self) -> Path:
        """Return the base path for language translation test data."""
        return Path("outputs") / "language_translation"

    def get_test_contexts(self, args) -> List[str]:
        """Return list of test contexts to run"""
        contexts = []
        for lan in args.languages:
            if lan not in CODE2LANGUAGE:
                raise ValueError(f"Unsupported language: {lan}")
            # Keep original behavior, but you can restrict by only passing --languages es
            contexts.append(f"tran_text_{lan}2en")
            contexts.append(f"tran_text_en2{lan}")
        INFO(f"Test contexts: {contexts}")
        return contexts

    async def prepare_prompts(self, test_id: Any, args) -> Tuple[List[Any], pd.DataFrame]:
        """Prepare prompts for the given context"""
        test_data = (
            self.exporter.read_table(
                QuestionsSchema,
                filters={"test_category": "language_translation", "test_id": test_id},
            )
            .sort_values(by="question_id", ascending=True)
            .reset_index(drop=True)
        )

        source_lang, target_lang = str(test_id).split("_")[-1].split("2")
        if test_data.empty:
            raise ValueError(f"No test data found for test_id: {test_id}")

        prompts: List[Any] = []
        for _, row in test_data.iterrows():
            prompts.append(
                self._create_benchmarking_prompt(
                    row=row,
                    system_content=self.settings.SYSTEM_MESSAGE.format(
                        source_lang=CODE2LANGUAGE[source_lang],
                        target_lang=CODE2LANGUAGE[target_lang],
                    ),
                    supports_system_messages=args.supports_system_messages,
                )
            )
        return prompts, test_data

    async def evaluate(
        self,
        context_data: Any,
        predictions: List[str],
        criteria: List[str] = ["rouge", "bleu", "meteor"],
    ) -> Tuple[List[str], Dict[str, List[float]], Dict[str, float]]:
        validations: Dict[str, List[float]] = defaultdict(list)
        for criterion in criteria:
            if criterion not in self.metrics:
                raise ValueError(f"Unsupported criterion: {criterion}")
            validations[criterion] = self.metrics[criterion](
                ground_truth=context_data["ground_truth_answer"].tolist(),
                predictions=predictions,
            )
        average_scores = {criterion: mean(scores) for criterion, scores in validations.items()}
        return predictions, validations, average_scores

    def _create_benchmarking_prompt(self, row, system_content, supports_system_messages=True):
        # kept from your file
        if supports_system_messages:
            messages = [{"role": "system", "content": system_content}]
        else:
            messages = [{"role": "user", "content": system_content}]
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": row["question"]}],
            }
        )
        return messages


class LanguageTranslationServiceConfig(BaseSpecialTestConfiguration):
    """
    This is the path you want: use a connector.translate(...) for inference.
    """

    def __init__(self, args, connector) -> None:
        self.args = args
        self.metric_assigner = CompositeTextSimilarity()
        self.metrics = {
            "rouge": self.metric_assigner.compute_rouge_score,
            "bleu": self.metric_assigner.compute_bleu_score,
            "meteor": self.metric_assigner.compute_meteor_score,
            # kept keys (placeholders)
            "bge_m3_similarity": self.metric_assigner.compute_similarity_scores,
            "ada_similarity": self.metric_assigner.compute_similarity_scores,
        }
        self.exporter = get_exporter(args)
        self.connector = connector

    def get_data(self, test_id) -> pd.DataFrame:
        return (
            self.exporter.read_table(
                QuestionsSchema,
                filters={"test_category": "language_translation", "test_id": test_id},
            )
            .sort_values(by="question_id", ascending=True)
            .reset_index(drop=True)
        )

    def get_test_contexts(self, args) -> List[str]:
        contexts = []
        for lan in args.languages:
            if lan not in CODE2LANGUAGE:
                raise ValueError(f"Unsupported language: {lan}")
            contexts.append(f"tran_text_{lan}2en")
            contexts.append(f"tran_text_en2{lan}")
        INFO(f"Test contexts: {contexts}")
        return contexts

    def get_predictions(self, context_data, context_id, **kwargs) -> List[str]:
        source_lang = context_id.split("_")[-1].split("2")[0]
        target_lang = context_id.split("_")[-1].split("2")[1]

        predictions: List[str] = []
        for _, row in context_data.iterrows():
            original_text = str(row["question"])
            translation = self.connector.translate(
                text=original_text,
                source_lang_id=source_lang,
                target_lang_id=target_lang,
            )
            predictions.append(translation)
        return predictions

    def evaluate(
        self,
        context_data: Any,
        predictions: List[str],
        criteria: List[str] = ["rouge", "bleu", "meteor"],
    ) -> Tuple[List[str], Dict[str, List[float]], Dict[str, float]]:
        validations: Dict[str, List[float]] = defaultdict(list)
        for criterion in criteria:
            if criterion not in self.metrics:
                raise ValueError(f"Unsupported criterion: {criterion}")
            validations[criterion] = self.metrics[criterion](
                ground_truth=context_data["ground_truth_answer"].tolist(),
                predictions=predictions,
            )
        average_scores = {criterion: mean(scores) for criterion, scores in validations.items()}
        return predictions, validations, average_scores


# ----------------------------
# CLI runner (self-contained)
# ----------------------------


def _write_sample_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
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
    df.to_csv(path, index=False)
    INFO(f"Wrote sample dataset to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/translation.csv")
    parser.add_argument("--test_id", type=str, default="tran_text_en2es")
    parser.add_argument("--languages", nargs="+", default=["es"])
    parser.add_argument("--supports_system_messages", action="store_true", default=False)

    parser.add_argument("--model_id", type=str, required=False, help="Your HF Hub model id, e.g. org/model")
    parser.add_argument("--device", type=str, default="auto", help="auto | -1 (cpu) | 0 (cuda:0) ...")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="For CausalLM/chat models: python format string with {source_lang} {target_lang} {text}",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="For chat models with tokenizer.chat_template: build prompt via apply_chat_template",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Enable transformers trust_remote_code for custom models",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=None,
        help="Fallback base model id (needed if your repo is a PEFT adapter or missing config/tokenizer)",
    )
    parser.add_argument(
        "--is_peft_adapter",
        action="store_true",
        default=False,
        help="Treat --model_id as a PEFT adapter and load it on top of --base_model_id",
    )

    parser.add_argument("--write_sample_data", action="store_true", default=False)
    args = parser.parse_args()

    if args.write_sample_data:
        _write_sample_csv(Path(args.data_path))
        return

    if not args.model_id:
        ERR("--model_id is required unless --write_sample_data is used")

    # Build connector and config
    connector = HuggingFaceHubTranslationConnector(
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        prompt_template=args.prompt_template,
        use_chat_template=args.use_chat_template,
        trust_remote_code=args.trust_remote_code,
        base_model_id=args.base_model_id,
        is_peft_adapter=args.is_peft_adapter,
    )
    config = LanguageTranslationServiceConfig(args=args, connector=connector)

    # Run one test_id (en->es by your request)
    context_id = args.test_id
    df = config.get_data(test_id=context_id)
    if df.empty:
        ERR(f"No rows found for test_id={context_id}. Check your CSV filters.")

    preds = config.get_predictions(context_data=df, context_id=context_id)
    _, per_example, averages = config.evaluate(context_data=df, predictions=preds)

    INFO(f"Ran {context_id} with model={args.model_id}")
    INFO(f"Average scores: {averages}")

    # Save outputs
    out_dir = Path("outputs") / "language_translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{context_id}__predictions.csv"
    out_df = df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(out_path, index=False)
    INFO(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

