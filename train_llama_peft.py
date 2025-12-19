from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import (
    IA3Config,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
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

    labels = []
    for i in range(len(input_ids)):
        # mask 掉 prompt 部分，只训练 target(Spanish)部分
        prompt_len = len(prompt_enc["input_ids"][i])
        lab = [-100] * prompt_len + input_ids[i][prompt_len:]
        lab = lab[: len(input_ids[i])]
        labels.append(lab)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _default_lora_targets(model_name: str) -> list[str]:
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


def _make_peft_config(
    method: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> Any:
    if method in {"lora", "qlora"}:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    if method == "ia3":
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            feedforward_modules=[m for m in target_modules if m in {"up_proj", "down_proj", "gate_proj"}],
        )
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on EN->ES sentence_pairs with PEFT.")
    parser.add_argument("--model", required=True, help="HF model name or local path (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--data", required=True, help="Path to JSON file: {'sentence_pairs':[...]} (your format)")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapters / checkpoints")

    # Hub / offline / cache
    parser.add_argument("--cache-dir", default=None, help="HF cache dir (optional)")
    parser.add_argument("--revision", default=None, help="Model revision (branch/tag/commit) if using Hub")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. If not set, reads env HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use fast tokenizer (default: false).",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Override HuggingFace Hub endpoint (e.g. internal mirror). Sets env HF_ENDPOINT.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not try to reach the internet; load only from local cache / local path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained (only if you trust the source).",
    )

    parser.add_argument("--method", choices=["lora", "qlora", "ia3"], default="lora", help="PEFT method")
    parser.add_argument("--target-modules", default=None, help="Comma-separated target module names (override defaults)")

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # LoRA 超参
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # QLoRA 量化配置
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    # 训练精度
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on A100/H100)")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.local_files_only:
        # Make Transformers/huggingface_hub behave offline.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    hf_token = args.hf_token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None

    # ---- Preflight info (helps debug offline/mirror issues) ----
    resolved_endpoint = os.getenv("HF_ENDPOINT") or "https://huggingface.co"
    model_path = args.model
    is_local_dir = os.path.isdir(model_path)
    print(
        "[preflight] "
        f"local_files_only={args.local_files_only} "
        f"HF_ENDPOINT={resolved_endpoint} "
        f"model_is_local_dir={is_local_dir}"
    )
    if not is_local_dir and resolved_endpoint == "https://huggingface.co":
        print(
            "[preflight] WARNING: model is a Hub id and HF_ENDPOINT is default. "
            "If you cannot access huggingface.co, set --hf-endpoint to your mirror, "
            "or use a local model directory with --local-files-only."
        )

    examples = _read_sentence_pairs_json(args.data)
    if not examples:
        raise RuntimeError(f"No examples found in {args.data}")
    ds = _build_dataset(examples)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=bool(args.use_fast),
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            token=hf_token,
        )
    except OSError as e:
        raise OSError(
            str(e)
            + "\n\n"
            + "Fix options:\n"
            + "- Use an internal HF mirror: pass --hf-endpoint https://YOUR_MIRROR (or set env HF_ENDPOINT)\n"
            + "- Or download/copy the model locally and pass --model /path/to/local/model --local-files-only\n"
            + "- If your network requires a proxy, set HTTPS_PROXY/HTTP_PROXY accordingly\n"
        ) from e
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = (
        [m.strip() for m in args.target_modules.split(",") if m.strip()]
        if args.target_modules
        else _default_lora_targets(args.model)
    )

    quant_config = None
    model_kwargs: dict[str, Any] = {}
    if args.method == "qlora":
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.bnb_4bit_compute_dtype]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            low_cpu_mem_usage=True,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            token=hf_token,
            **model_kwargs,
        )
    except OSError as e:
        raise OSError(
            str(e)
            + "\n\n"
            + "Fix options:\n"
            + "- Use an internal HF mirror: pass --hf-endpoint https://YOUR_MIRROR (or set env HF_ENDPOINT)\n"
            + "- Or download/copy the model locally and pass --model /path/to/local/model --local-files-only\n"
            + "- If your network requires a proxy, set HTTPS_PROXY/HTTP_PROXY accordingly\n"
        ) from e

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.method == "qlora":
        model = prepare_model_for_kbit_training(model)

    peft_cfg = _make_peft_config(
        args.method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)

    tokenized = ds.map(
        lambda b: _tokenize_and_mask(b, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

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
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        report_to=[],
        optim="paged_adamw_8bit" if args.method == "qlora" else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 adapter + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 可选：如果你想导出合并后的全量模型（更大、但推理更方便），取消注释：
    # if isinstance(model, PeftModel):
    #     merged = model.merge_and_unload()
    #     merged.save_pretrained(args.output_dir + "_merged")


if __name__ == "__main__":
    main()

