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
# 🔧 CONFIGURATION (原来通过 argparse 传的，全放在这里)
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# 从 Hugging Face Hub 加载数据集（不再从本地读取）
DATASET_ID = "jiejiewuistaken/COSOP_enes"

# OUTPUT_DIR = "/outputs"  # HF Jobs 持久化目录
OUTPUT_DIR = "./outputs"  # HF Jobs 持久化目录

# 训练参数
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

# LoRA 参数
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

# push 到 Hub 的 repo
HUB_REPO_ID = "jiejiewuistaken/finetuned-mt-model"

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


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(SEED)

    # 直接从 Hub 加载数据
    ds_dict = load_dataset(DATASET_ID)
    if "train" in ds_dict:
        dataset = ds_dict["train"]
    else:
        # 兜底：如果没有 train split，就取第一个 split
        first_split = next(iter(ds_dict.keys()))
        dataset = ds_dict[first_split]

    if not dataset:
        raise RuntimeError(f"No training examples found in {DATASET_ID}")

    # 需要列名为 en/es（与后续 tokenize 逻辑保持一致）
    cols = set(dataset.column_names)
    if not {"en", "es"}.issubset(cols):
        raise ValueError(
            f"Dataset {DATASET_ID} must contain columns ['en', 'es'], got: {dataset.column_names}"
        )

    hf_token = os.environ.get("HF_TOKEN")

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
        lambda b: _tokenize_and_mask(b, tokenizer=tokenizer, max_length=MAX_LENGTH),
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
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA adapter + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 推送到 Hugging Face Hub
    trainer.push_to_hub(repo_id=HUB_REPO_ID)


if __name__ == "__main__":
    main()

