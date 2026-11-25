#!/usr/bin/env bash
# Run several vLLM evals one after another on the pod. Usage:
#   bash scripts/eval_chain.sh "<model> <tag> <split> [revision]" ...
set -uo pipefail
cd /workspace/gsm8k-rl && export HF_HOME=/workspace/hf && source /workspace/venv/bin/activate
for job in "$@"; do
  set -- $job
  rev=${4:+--revision $4}
  python src/eval_vllm.py --model "$1" --tag "$2" --split "$3" --thinking $rev 2>&1 | grep -a -E "acc |saved ->|Error|Traceback"
done
