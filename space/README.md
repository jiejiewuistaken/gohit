---
title: Translation (Base vs Fine-tuned)
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---

## What this Space does

- **Switch button**: swap translation direction (From/To)
- **Input**: the sentence to translate
- **Two outputs**: base model output vs fine-tuned (LoRA adapter) output

## Setup (Space Variables / Secrets)

Set these in your Space settings:

- **Secrets**
  - `HF_TOKEN`: required if the base model is gated (e.g. Llama) or your adapter repo is private
- **Variables**
  - `BASE_MODEL_ID`: default `meta-llama/Llama-3.1-8B-Instruct`
  - `ADAPTER_REPO_ID`: your LoRA adapter repo id

## Notes

- For Llama-3.1-8B you usually need a **GPU Space**.
- This app loads the base model once and uses `disable_adapter()` to produce the **base** output without loading a second copy.

