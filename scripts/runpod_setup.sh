#!/usr/bin/env bash
# One-time setup on a fresh RunPod GPU pod (PyTorch template, 1x H100).
# Run ON THE POD after copying the project to /workspace/gsm8k-rl:
#     bash scripts/runpod_setup.sh
set -euo pipefail
cd /workspace/gsm8k-rl

nvidia-smi --query-gpu=name,memory.total --format=csv

# Same TRL version as the Mac env; the [vllm] extra pulls a vLLM build TRL supports.
pip install -q "trl[vllm]==1.18.2" peft pyyaml pytest
export HF_HOME=/workspace/hf   # model downloads land on the network volume (survive pod stops)

# data/*.jsonl is git-ignored; rebuild it if it wasn't copied (seed 42 = identical splits).
[ -f data/train.jsonl ] || python src/make_splits.py

python -m pytest tests/ -q
python -c "import torch, vllm, trl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| cuda', torch.cuda.is_available())"
echo "Setup OK. Next, inside tmux so it survives disconnects (tmux new -s train):"
echo "  cd /workspace/gsm8k-rl && export HF_HOME=/workspace/hf"
echo "  python src/profile_difficulty.py configs/grpo_qwen3_17b.yaml 2>&1 | tee profile.log"
echo "  python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=qwen3-17b-pilot max_steps=20 eval_n=null 2>&1 | tee pilot.log"
