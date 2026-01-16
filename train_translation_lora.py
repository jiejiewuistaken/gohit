# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "transformers>=4.40",
#   "datasets",
#   "peft",
#   "accelerate",
#   "sentencepiece",
#   "protobuf<5",
# ]
# ///

import os
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfFolder, login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Language direction (override via env vars if needed)
SOURCE_LANG = os.environ.get("SOURCE_LANG", "en").lower()
TARGET_LANG = os.environ.get("TARGET_LANG", "es").lower()

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}
ALLOWED_LANGS = set(LANGUAGE_NAMES.keys())

# Load dataset from Hugging Face Hub
DATASET_ID = "jiejiewuistaken/IFAD-mt-en-es-v0.1"

OUTPUT_DIR = "./outputs"  # HF Jobs persistent directory

# training args
MAX_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1.0
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2
SEED = 42

# LoRA params
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# push to Hub repo (override via env var for different directions)
HUB_REPO_ID = os.environ.get(
    "HUB_REPO_ID",
    "ifadaiml/Llama-3.1-8B-Instruct-IFAD-mt-cosop-es-en-v0.1",
)

# ============================================================


def _format_prompt(source_text: str, *, source_lang: str, target_lang: str) -> str:
    source_name = LANGUAGE_NAMES[source_lang]
    target_name = LANGUAGE_NAMES[target_lang]
    return (
        f"Translate {source_name} to {target_name}.\n\n"
        f"{source_name}: {source_text.strip()}\n"
        f"{target_name}:"
    )


def _tokenize_and_mask(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
    max_length: int,
    source_lang: str,
    target_lang: str,
) -> dict[str, Any]:
    prompts = [
        _format_prompt(text, source_lang=source_lang, target_lang=target_lang)
        for text in batch[source_lang]
    ]
    targets = [" " + text.strip() for text in batch[target_lang]]

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

    labels = []
    for i, ids in enumerate(full_enc["input_ids"]):
        prompt_len = len(prompt_enc["input_ids"][i])
        lab = [-100] * prompt_len + ids[prompt_len:]
        labels.append(lab[: len(ids)])

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def _validate_language_pair(source_lang: str, target_lang: str) -> None:
    if source_lang not in ALLOWED_LANGS or target_lang not in ALLOWED_LANGS:
        raise ValueError(
            "Language codes must be one of "
            f"{sorted(ALLOWED_LANGS)}. Got: {source_lang}, {target_lang}"
        )
    if source_lang == target_lang:
        raise ValueError("SOURCE_LANG and TARGET_LANG must be different.")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(SEED)

    _validate_language_pair(SOURCE_LANG, TARGET_LANG)

    # Token resolution:
    # - Prefer explicit env vars (common in jobs/CI)
    # - Fallback to locally cached token (huggingface-cli login)
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

    # Ensure hub auth is available globally as well.
    # This helps for any internal hub calls that don't pass `token=...` through.
    login(token=hf_token, add_to_git_credential=False)

    ds_dict = load_dataset(
        DATASET_ID,
        token=hf_token,
    )
    if "train" in ds_dict:
        dataset = ds_dict["train"]
    else:
        # Fallback: use the first split if no "train" split exists.
        first_split = next(iter(ds_dict.keys()))
        dataset = ds_dict[first_split]

    if not dataset:
        raise RuntimeError(f"No training examples found in {DATASET_ID}")

    # Expect columns matching SOURCE_LANG / TARGET_LANG
    cols = set(dataset.column_names)
    if not {SOURCE_LANG, TARGET_LANG}.issubset(cols):
        raise ValueError(
            f"Dataset {DATASET_ID} must contain columns "
            f"[{SOURCE_LANG!r}, {TARGET_LANG!r}], got: {dataset.column_names}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)

    tokenized = dataset.map(
        lambda b: _tokenize_and_mask(
            b,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            source_lang=SOURCE_LANG,
            target_lang=TARGET_LANG,
        ),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
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
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Push to Hugging Face Hub
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
