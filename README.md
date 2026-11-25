# gsm8k-rl

**Training a small Qwen model with reinforcement learning (RLVR / GRPO) to top the ≤6B GSM8K leaderboard, after first checking why the current leaders score what they score.**

GSM8K is a set of grade-school math word problems. A model must solve each one and give a single number. Only the final number is graded.

## Results so far

Same strict test setup for every model, 500 held-out questions (details in [EVAL_RESULTS.md](EVAL_RESULTS.md)):

| Model | Size | Best setting | Our score | Leaderboard claim |
|---|---|---|---|---|
| granite-4.1-3b | 3.4B | zero-shot | 94.2 | 86.88 |
| Phi-3-mini-4k | 3.8B | zero-shot | 89.6 | 85.7 |
| Phi-3.5-mini | 3.8B | zero-shot | 87.0 | 86.2 |
| **Qwen3-1.7B** (our base) | 1.7B | zero-shot | **84.0** | – |
| Qwen3-0.6B | 0.6B | thinking | 77.2 | – |
| Axonry Qwen3.5-0.8B | 0.87B | thinking | 76.8 | 84.03 |

Key findings:
1. **Leaderboard scores are not comparable.** They are self-reported, and each team uses its own test setup.
2. **Few-shot examples hurt every model.** Models copy the short example style and reason less.
3. **Small models break on long problems**: 5+ step problems cost them 20–25 points.
4. **75% of training questions are "too easy"** for Qwen3-1.7B, so GRPO would learn nothing from them. We filter for questions the model *sometimes* gets right.

## How it works

```
GSM8K train ──► difficulty profiling ──► learnable questions ──► GRPO (TRL + vLLM) ──► eval (MLX)
                (8 answers / question)    (right 1–7 of 8)        reward = correct "#### N"
```

- **One prompt, one grader.** Training and evaluation share `src/prompts.py` and `src/grading.py`, so the model is trained on exactly what we measure.
- **Verifiable reward.** The reward is 1.0 for a correct `#### N`, 0.5 for a correct `#### N.0` (some graders reject that), and 0 otherwise. No human labels and no reward model.
- **Reproducible.** Every model is pinned to an exact Hugging Face commit, and every training run saves its config plus the git commit that produced it.

## Repository layout

| Path | What |
|---|---|
| [`EVAL_RESULTS.md`](EVAL_RESULTS.md) | **All evaluation results and analysis** |
| `src/make_splits.py` | Builds fixed splits: train 6,973 / dev 500 (held out from train) / test 1,319 |
| `src/prompts.py` | The single prompt used everywhere |
| `src/grading.py` | Answer extraction and correctness check |
| `src/eval.py` | Evaluation on a Mac (MLX), saves every answer for analysis |
| `src/sweep.py`, `src/aggregate.py` | Run many eval settings from a YAML file, then summarize |
| `src/rewards.py` | RL reward functions (built on `grading.py`) |
| `src/profile_difficulty.py` | Samples 8 answers per train question and keeps the learnable ones |
| `src/train_grpo.py` | GRPO training with TRL; `key=value` overrides on the command line |
| `configs/` | Eval sweeps (`phase1.yaml`, `qwen3_17b_baseline.yaml`) and training configs (`grpo_qwen3_17b.yaml`) |
| `scripts/runpod_setup.sh` | One-command setup on a cloud GPU pod |
| `tests/` | Reward function tests |
| `results/dev/summary.csv` | Summary of every eval run |

Not in the repo (git-ignored): data files (rebuilt by `make_splits.py`), per-question result files, model weights, and checkpoints.

## Quick start

### Evaluate on a Mac (Apple Silicon)

```bash
conda env create -f environment.yml && conda activate gsm8k
python src/make_splits.py                                    # downloads GSM8K, writes data/*.jsonl
python src/eval.py --model Qwen/Qwen3-1.7B --split dev --n 50
python src/sweep.py configs/phase1.yaml && python src/aggregate.py
```

### Train on a cloud GPU (1x H100 80GB)

```bash
bash scripts/runpod_setup.sh                                  # venv + TRL + vLLM, runs tests
export HF_HOME=/workspace/hf && source /workspace/venv/bin/activate
python src/profile_difficulty.py configs/grpo_qwen3_17b.yaml  # -> data/train_filtered.jsonl
python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=pilot max_steps=20 eval_n=null
python src/train_grpo.py configs/grpo_qwen3_17b.yaml          # run A
python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=qwen3-17b-grpo-B \
    epsilon_high=0.28 scale_rewards=none                      # run B (ablation)
```

### Tests

```bash
python -m pytest tests/ -q
```

## Rules we follow

1. Develop on **dev**. Use **test** only for final reported numbers.
2. Every reported number records its model version, prompt, decoding settings, and grading rule.
3. Never train on evaluation sets (dev, test, GSM-Plus, GSM-Symbolic).

## Roadmap

- [x] Phase 1: Re-test leaderboard models under one setup ([results](EVAL_RESULTS.md))
- [x] Choose the base model: Qwen3-1.7B (thinking)
- [ ] Phase 2: Difficulty profiling, GRPO pilot, runs A and B (ablation)
- [ ] Phase 3: Before/after analysis on dev, then a single test-set run, plus GSM-Plus to check generalization
- [ ] Phase 4: Publish the model on Hugging Face with a full method card

## Credits

- Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT)
- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0)
- Libraries: [MLX](https://github.com/ml-explore/mlx), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm)
