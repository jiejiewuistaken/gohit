import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    return str(o)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return s or None
    except FileNotFoundError:
        return None


def read_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def try_get_git_sha() -> str | None:
    # Avoid importing subprocess unless needed (helps minimal environments).
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return sha or None
    except Exception:
        return None


def resolve_job_context() -> dict[str, Any]:
    """
    Best-effort job context resolver.

    HF Jobs sometimes injects identifiers via env vars (unknown/unstable names),
    so we collect common candidates. If none exist, we also support injecting
    a file (e.g. 'job_id.txt' or 'hf_job_metadata.json') into the working dir.
    """
    env_priority = [
        "HF_JOB_ID",
        "HUGGINGFACE_JOB_ID",
        "HUGGINGFACE_JOB_RUN_ID",
        "HF_RUN_ID",
        "JOB_ID",
        "RUN_ID",
        "SLURM_JOB_ID",
        "AWS_BATCH_JOB_ID",
    ]

    ctx: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "cwd": os.getcwd(),
        "run_tag": os.environ.get("RUN_TAG") or os.environ.get("JOB_TAG") or None,
        "git_sha": try_get_git_sha(),
        "env_job_ids": {},
        "job_id": None,
        "job_id_source": None,
    }

    for k in env_priority:
        v = os.environ.get(k)
        if v:
            ctx["env_job_ids"][k] = v

    for k in env_priority:
        v = os.environ.get(k)
        if v:
            ctx["job_id"] = v
            ctx["job_id_source"] = f"env:{k}"
            break

    meta_paths: list[str] = []
    if os.environ.get("HF_JOB_METADATA_PATH"):
        meta_paths.append(os.environ["HF_JOB_METADATA_PATH"])
    meta_paths.extend(
        [
            os.path.join(os.getcwd(), "hf_job_metadata.json"),
            os.path.join(os.getcwd(), "job_metadata.json"),
            os.path.join(os.getcwd(), "job_id.txt"),
        ]
    )

    for p in meta_paths:
        j = read_json_file(p)
        if j:
            ctx["job_metadata_file"] = p
            ctx["job_metadata"] = j
            # Try typical keys
            for key in ("job_id", "id", "hf_job_id", "run_id"):
                if ctx["job_id"] is None and isinstance(j.get(key), str) and j.get(key):
                    ctx["job_id"] = j[key]
                    ctx["job_id_source"] = f"file:{p}:{key}"
            break

    if ctx["job_id"] is None:
        txt = read_text_file(os.path.join(os.getcwd(), "job_id.txt"))
        if txt:
            ctx["job_id"] = txt
            ctx["job_id_source"] = "file:job_id.txt"

    # Convenience: include submit time if provided externally.
    if os.environ.get("JOB_SUBMIT_TIME_UNIX"):
        try:
            ctx["job_submit_time_unix"] = float(os.environ["JOB_SUBMIT_TIME_UNIX"])
        except ValueError:
            ctx["job_submit_time_unix"] = os.environ["JOB_SUBMIT_TIME_UNIX"]
    else:
        ctx["job_submit_time_unix"] = None

    # Always include a coarse "process start" time to help correlate logs.
    ctx["process_start_time_unix"] = time.time()
    return ctx

