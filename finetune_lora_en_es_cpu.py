import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass(frozen=True)
class Example:
    en: str
    es: str


def _read_sentence_pairs_json(path: str) -> list[Example]:
    """
    Accepts:
    - a single JSON file shaped like {"sentence_pairs":[{...}, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    pairs = obj.get("sentence_pairs", [])
    out: list[Example] = []
    for p in pairs:
        s1 = (p.get("sentence_1") or {}).get("content", "")
        s2 = (p.get("sentence_2") or {}).get("content", "")
        s1 = (s1 or "").strip()
        s2 = (s2 or "").strip()
        if not s1 or not s2:
            continue
        out.append(Example(en=s1, es=s2))
    return out


def _build_dataset(examples: list[Example]) -> Dataset:
    return Dataset.from_list([{"en": e.en, "es": e.es} for e in examples])


def _format_prompt(en: str) -> str:
    # 简单稳定的指令格式（适用于 base/inst 模型）
    return "Translate English to Spanish.\n\nEnglish: " + en.strip() + "\nSpanish:"


def _tokenize_and_mask(
    batch: dict[str, list[str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    prompts = [_format_prompt(en) for en in batch["en"]]
    targets = [(" " + es.strip()) for es in batch["es"]]  # 确保目标和冒号之间有空格

    prompt_enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    full_texts = [p + t for p, t in zip(prompts, targets)]
    full_enc = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    labels: list[list[int]] = []
    for i in range(len(input_ids)):
        # mask 掉 prompt 部分，只训练 target(Spanish)部分
        prompt_len = len(prompt_enc["input_ids"][i])
        lab = [-100] * prompt_len + input_ids[i][prompt_len:]
        lab = lab[: len(input_ids[i])]
        labels.append(lab)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _default_lora_targets(_: str) -> list[str]:
    # LLaMA/Meta-Llama/Qwen-LLaMA 风格常用 target modules
    # 注意：不同模型可能模块名不同；你可以用 CLI 覆盖
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only LoRA fine-tune: EN->ES sentence_pairs (your JSON format)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name or local path (e.g. meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to JSON file: {'sentence_pairs':[...]} (your format)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for LoRA adapter + tokenizer",
    )

    parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated target module names (override defaults)",
    )
    parser.add_argument("--max-length", type=int, default=512)

    # CPU 默认值：小 batch + 少 worker，避免把机器打满
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA 超参
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # HF token（可选；也可以用 huggingface-cli login）
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If not set, uses env HF_TOKEN or local login cache.",
    )

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    examples = _read_sentence_pairs_json(args.data)
    if not examples:
        raise RuntimeError(f"No training examples found in {args.data}")
    ds = _build_dataset(examples)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = (
        [m.strip() for m in args.target_modules.split(",") if m.strip()]
        if args.target_modules
        else _default_lora_targets(args.model)
    )

    # CPU-only：不做量化/半精度/device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)

    tokenized = ds.map(
        lambda b: _tokenize_and_mask(b, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    # 让 collator 负责 padding input_ids/attention_mask/labels（labels 用 -100 pad）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        optim="adamw_torch",
        # CPU-only：明确禁用混合精度
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

    # 保存 adapter + tokenizer（只保存 LoRA，不会导出全量模型）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
