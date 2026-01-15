#!/usr/bin/env bash
set -euo pipefail

# This script is meant to be run on YOUR machine when you submit a Hugging Face Job.
# It captures the CLI output, tries to extract the printed job id, and writes:
# - ./job_id.txt
# - ./job_metadata.json
#
# Usage:
#   ./submit_hf_job_and_log.sh huggingface-cli jobs run ...your args...
#
# Notes:
# - Some HF Jobs environments DO NOT expose the job id inside the training container.
#   In that case, your Python code can't "magically" know the id; you must record it
#   on submit (this script), or inject a stable RUN_TAG into the job environment.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <submit-command...>" >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-run-$(date +%Y%m%d-%H%M%S)-$RANDOM}"
export RUN_TAG
export JOB_SUBMIT_TIME_UNIX="$(date +%s)"

LOG_FILE="${LOG_FILE:-hf_job_submit.log}"

echo "RUN_TAG=$RUN_TAG"
echo "Logging submit output to: $LOG_FILE"

set +e
OUTPUT="$("$@" 2>&1 | tee "$LOG_FILE")"
STATUS=${PIPESTATUS[0]}
set -e

# Best-effort job id extraction from common patterns.
# We try a few regexes and take the first match.
JOB_ID=""

try_extract() {
  local re="$1"
  local m
  m="$(printf "%s" "$OUTPUT" | perl -ne "if (/$re/) { print \$1; exit 0 }" 2>/dev/null)"
  if [[ -n "${m:-}" ]]; then
    JOB_ID="$m"
    return 0
  fi
  return 1
}

# Patterns seen across CLIs vary; keep this list broad.
try_extract 'Job ID[:\s]+([A-Za-z0-9._-]+)' || true
try_extract 'job id[:\s]+([A-Za-z0-9._-]+)' || true
try_extract 'job[:\s]+([A-Za-z0-9._-]{8,})' || true
try_extract 'id[:\s]+([A-Za-z0-9._-]{8,})' || true

if [[ -n "$JOB_ID" ]]; then
  echo "$JOB_ID" > job_id.txt
  echo "Extracted JOB_ID=$JOB_ID (written to job_id.txt)"
else
  echo "Warning: could not extract job id from submit output." >&2
fi

cat > job_metadata.json <<EOF
{
  "run_tag": "$(printf "%s" "$RUN_TAG" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')",
  "job_id": "$(printf "%s" "$JOB_ID" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')",
  "submit_status": $STATUS,
  "submit_time_unix": $JOB_SUBMIT_TIME_UNIX,
  "log_file": "$(printf "%s" "$LOG_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')"
}
EOF

echo "Wrote job_metadata.json"
exit $STATUS

