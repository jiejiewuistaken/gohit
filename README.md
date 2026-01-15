# LoRA grid search + local logging (HF Jobs friendly)

This repo contains:

- `train_lora_grid.py`: LoRA grid search training script + **local JSONL logging**
- `hf_job_logging.py`: helper for appending JSONL + best-effort job-id resolution
- `submit_hf_job_and_log.sh`: optional submit wrapper to record the **CLI-printed** HF Jobs job id

## What gets logged locally

`train_lora_grid.py` appends to:

- `outputs/experiment_log.jsonl`

Events include:

- `session_started` / `session_finished`
- `run_started` / `run_finished` / `run_failed`

Each run logs:

- LoRA params + learning rate
- `eval_loss` curve (per eval step/epoch)
- `best_eval_loss`, `best_epoch`, `best_checkpoint`

## About Hugging Face Jobs job id

There are 2 cases:

1. **HF Jobs injects a job id into the container** (via env var).  
   Then `resolve_job_context()` will capture it automatically (e.g. `HF_JOB_ID`, `JOB_ID`, ...).

2. **HF Jobs does NOT inject job id into the container** (common).  
   Then the Python training code cannot know it, because it only appears on the submitter CLI.

Workarounds:

- **Use a stable tag** you control:

```bash
export RUN_TAG="mt-en-es-$(date +%Y%m%d-%H%M%S)"
```

and pass it into the job environment (recommended). The script will log `run_tag`.

- **Record the CLI output at submit time**:

```bash
chmod +x submit_hf_job_and_log.sh
./submit_hf_job_and_log.sh huggingface-cli jobs run ...your args...
```

This writes `job_id.txt` and `job_metadata.json` next to the submit command.

