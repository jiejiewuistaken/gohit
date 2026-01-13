"""
LoRA grid search training script (Llama-3.1) + local JSONL logging.

What this adds vs. the plain Trainer loop:
- Writes one JSON line per run to `outputs/experiment_log.jsonl`
- Records params + validation (eval_loss) curve + best checkpoint/epoch
- Best-effort captures Hugging Face Jobs job id from env/file (if available)

Tip: If HF Jobs does NOT inject a job id into the container, set a stable tag:
  export RUN_TAG="my-run-$(date +%Y%m%d-%H%M%S)"
and you'll still be able to correlate logs with the job submission output.
"""

import inspect
import os
import traceback
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfFolder, login
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from hf_job_logging import append_jsonl, resolve_job_context

# ============================================================
#  CONFIGURATION
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# load dataset from HF Hub
DATASET_ID = "jiejiewuistaken/IFAD-mt-en-es-v0.1"

OUTPUT_ROOT_DIR = "./outputs"
EXPERIMENT_LOG_PATH = os.path.join(OUTPUT_ROOT_DIR, "experiment_log.jsonl")

# data / split
MAX_LENGTH = 512
VAL_RATIO = 0.05
SEED = 42

# training base args
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 5.0  # early stopping decides optimal epoch
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2

# early stopping（based on eval_loss）
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0

# LoRA target modules
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# grid search space
LORA_R_GRID = [16]
LORA_ALPHA_GRID = [32]
LORA_DROPOUT_GRID = [0.05]
LEARNING_RATE_GRID = [1e-4, 2e-4]

# push to Hub
PUSH_BEST_TO_HUB = True
HUB_REPO_ID = "ifadaiml/Llama-3.1-8B-Instruct-IFAD-mt-en-es-v0.2"

# ============================================================


def _format_prompt(en: str) -> str:
    return "Translate English to Spanish.\n\nEnglish: " + en.strip() + "\nSpanish:"


def _tokenize_and_mask(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    prompts = [_format_prompt(en) for en in batch["en"]]
    targets = [" " + es.strip() for es in batch["es"]]

    prompt_enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    full_enc = tokenizer(
        [p + t for p, t in zip(prompts, targets)],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    labels: list[list[int]] = []
    for i, ids in enumerate(full_enc["input_ids"]):
        prompt_len = len(prompt_enc["input_ids"][i])
        lab = [-100] * prompt_len + ids[prompt_len:]
        labels.append(lab[: len(ids)])

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def _resolve_hf_token() -> str:
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
        or HfFolder.get_token()
    )
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
            "to a token that has access to the gated model and dataset."
        )
    return hf_token


def _make_training_arguments(**kwargs: Any) -> TrainingArguments:
    """
    Compatibility wrapper across transformers versions.

    - `evaluation_strategy` was renamed to `eval_strategy` in newer versions.
    - We only pass kwargs that exist in the installed `TrainingArguments` signature.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    accepted = set(sig.parameters.keys())

    # Handle renamed fields
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in accepted:
        if "eval_strategy" in accepted and "eval_strategy" not in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
        else:
            kwargs.pop("evaluation_strategy", None)

    # If both are present, prefer the one supported by this version.
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
        kwargs.pop("eval_strategy", None)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return TrainingArguments(**filtered)


def _load_and_split_dataset(*, hf_token: str) -> tuple[Dataset, Dataset]:
    ds_dict: DatasetDict = load_dataset(
        DATASET_ID,
        token=hf_token,
    )
    if "train" in ds_dict:
        base = ds_dict["train"]
    else:
        first_split = next(iter(ds_dict.keys()))
        base = ds_dict[first_split]

    if not base:
        raise RuntimeError(f"No training examples found in {DATASET_ID}")

    cols = set(base.column_names)
    if not {"en", "es"}.issubset(cols):
        raise ValueError(
            f"Dataset {DATASET_ID} must contain columns ['en', 'es'], got: {base.column_names}"
        )

    split = base.train_test_split(test_size=VAL_RATIO, seed=SEED, shuffle=True)
    return split["train"], split["test"]


def _best_epoch_from_log_history(log_history: list[dict[str, Any]]) -> float | None:
    best = None
    for row in log_history:
        if "eval_loss" not in row:
            continue
        loss = row.get("eval_loss")
        epoch = row.get("epoch")
        if loss is None or epoch is None:
            continue
        if isinstance(loss, float) and (loss != loss):  # NaN
            continue
        if best is None or loss < best["eval_loss"]:
            best = {"eval_loss": float(loss), "epoch": float(epoch)}
    return None if best is None else best["epoch"]


def _extract_eval_curve(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    curve: list[dict[str, float]] = []
    for row in log_history:
        if "eval_loss" not in row:
            continue
        loss = row.get("eval_loss")
        epoch = row.get("epoch")
        if loss is None or epoch is None:
            continue
        try:
            curve.append({"epoch": float(epoch), "eval_loss": float(loss)})
        except Exception:
            continue
    # Keep chronological order, but stable sort by epoch just in case.
    curve.sort(key=lambda x: x["epoch"])
    return curve


@dataclass(frozen=True)
class GridConfig:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float

    def run_name(self) -> str:
        return (
            f"r{self.lora_r}_a{self.lora_alpha}_d{self.lora_dropout}_lr{self.learning_rate}"
        )


def _train_one(
    cfg: GridConfig,
    *,
    hf_token: str,
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: Any,
    job_ctx: dict[str, Any],
) -> dict[str, Any]:
    run_dir = os.path.join(OUTPUT_ROOT_DIR, cfg.run_name())
    os.makedirs(run_dir, exist_ok=True)

    append_jsonl(
        EXPERIMENT_LOG_PATH,
        {
            "event": "run_started",
            "run_name": cfg.run_name(),
            "run_dir": run_dir,
            "config": cfg,
            "job": job_ctx,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    # Recommended for training, avoids warnings and sometimes memory spikes
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)

    tokenized_train = train_ds.map(
        lambda b: _tokenize_and_mask(b, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=train_ds.column_names,
        desc=f"Tokenizing train ({cfg.run_name()})",
    )
    tokenized_val = val_ds.map(
        lambda b: _tokenize_and_mask(b, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=val_ds.column_names,
        desc=f"Tokenizing val ({cfg.run_name()})",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    train_args = _make_training_arguments(
        output_dir=run_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=cfg.learning_rate,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to=[],
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        push_to_hub=False,  # grid search middle don't push
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ],
    )

    trainer.train()

    log_history = list(trainer.state.log_history or [])
    best_epoch = _best_epoch_from_log_history(log_history)
    eval_curve = _extract_eval_curve(log_history)
    best_metric = trainer.state.best_metric  # eval_loss（smaller the better）
    best_ckpt = trainer.state.best_model_checkpoint

    # store adapter + tokenizer（load_best_model_at_end optim weight）
    trainer.model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)

    del model
    torch.cuda.empty_cache()

    result = {
        "config": cfg,
        "run_dir": run_dir,
        "best_eval_loss": best_metric,
        "best_epoch": best_epoch,
        "best_checkpoint": best_ckpt,
        "eval_curve": eval_curve,
    }

    append_jsonl(
        EXPERIMENT_LOG_PATH,
        {
            "event": "run_finished",
            "run_name": cfg.run_name(),
            "run_dir": run_dir,
            "result": result,
            "job": job_ctx,
        },
    )

    return result


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(SEED)

    job_ctx = resolve_job_context()
    append_jsonl(EXPERIMENT_LOG_PATH, {"event": "session_started", "job": job_ctx})

    hf_token = _resolve_hf_token()
    login(token=hf_token, add_to_git_credential=False)

    train_ds, val_ds = _load_and_split_dataset(hf_token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    grid: list[GridConfig] = [
        GridConfig(lora_r=r, lora_alpha=a, lora_dropout=d, learning_rate=lr)
        for r in LORA_R_GRID
        for a in LORA_ALPHA_GRID
        for d in LORA_DROPOUT_GRID
        for lr in LEARNING_RATE_GRID
    ]

    results: list[dict[str, Any]] = []
    for i, cfg in enumerate(grid, start=1):
        print(f"\n==== Grid run {i}/{len(grid)}: {cfg.run_name()} ====")
        try:
            res = _train_one(
                cfg,
                hf_token=hf_token,
                train_ds=train_ds,
                val_ds=val_ds,
                tokenizer=tokenizer,
                job_ctx=job_ctx,
            )
            results.append(res)
            print(
                f"-> done: best_eval_loss={res['best_eval_loss']}, best_epoch={res['best_epoch']}, "
                f"best_checkpoint={res['best_checkpoint']}"
            )
        except Exception as e:
            append_jsonl(
                EXPERIMENT_LOG_PATH,
                {
                    "event": "run_failed",
                    "run_name": cfg.run_name(),
                    "run_dir": os.path.join(OUTPUT_ROOT_DIR, cfg.run_name()),
                    "config": cfg,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "job": job_ctx,
                },
            )
            raise

    # best config（min eval_loss）
    def _key(r: dict[str, Any]) -> float:
        m = r.get("best_eval_loss")
        return float("inf") if m is None else float(m)

    best = min(results, key=_key)
    best_cfg: GridConfig = best["config"]
    print("\n================ BEST CONFIG ================")
    print(f"best_config: {best_cfg.run_name()}")
    print(f"best_eval_loss: {best['best_eval_loss']}")
    print(f"best_epoch: {best['best_epoch']}")
    print(f"best_checkpoint: {best['best_checkpoint']}")
    print(f"best_run_dir: {best['run_dir']}")

    append_jsonl(
        EXPERIMENT_LOG_PATH,
        {
            "event": "session_finished",
            "best_config": best_cfg,
            "best_result": best,
            "job": job_ctx,
        },
    )

    # push best run
    if PUSH_BEST_TO_HUB:
        # reload base model + best adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        model = PeftModel.from_pretrained(base_model, best["run_dir"])

        train_args = _make_training_arguments(
            output_dir=best["run_dir"],
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=best_cfg.learning_rate,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="no",
            save_strategy="no",
            seed=SEED,
            report_to=[],
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            push_to_hub=True,
            hub_model_id=HUB_REPO_ID,
            hub_token=hf_token,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            tokenizer=tokenizer,
        )
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

