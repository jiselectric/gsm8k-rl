# gsm8k-rl

**Can reinforcement learning (RLVR / GRPO) push a small Qwen model to the top of the ≤6B GSM8K leaderboard? We re-tested the leaders under one strict setup, then trained Qwen3-1.7B with GRPO on an H100.**

**Short answer:** the *untrained* Qwen3-1.7B (thinking mode) already scores **90.4 on the GSM8K test set**, above every ≤6B leaderboard claim. Four GRPO runs did **not** improve on it: three on GSM8K, plus one on harder MATH problems. The highest learning rate made it worse. This repo documents both the evaluation and that negative result.

GSM8K is a set of grade-school math word problems. A model must solve each one and give a single number. Only the final number is graded.

## Evaluation results

Same strict test setup for every model, 500 held-out questions (details in [EVAL_RESULTS.md](EVAL_RESULTS.md)):

| Model | Size | Best setting | Our score | Leaderboard claim |
|---|---|---|---|---|
| granite-4.1-3b | 3.4B | zero-shot | 94.2 | 86.88 |
| Phi-3-mini-4k | 3.8B | zero-shot | 89.6 | 85.7 |
| Phi-3.5-mini | 3.8B | zero-shot | 87.0 | 86.2 |
| **Qwen3-1.7B** (our base) | 1.7B | thinking | **91.4** (test: **90.4**) | – |
| Qwen3-0.6B | 0.6B | thinking | 77.2 | – |
| Axonry Qwen3.5-0.8B | 0.87B | thinking | 76.8 | 84.03 |

Key findings:
1. **Leaderboard scores are not comparable.** They are self-reported, and each team uses its own test setup.
2. **Few-shot examples hurt every model.** Models copy the short example style and reason less.
3. **Small models break on long problems**: 5+ step problems cost them 20–25 points.
4. **82% of training questions are "too easy"** for Qwen3-1.7B, so GRPO would learn nothing from them. We filter for questions the model *sometimes* gets right (1,029 of 6,973).

## GRPO results

| Run | Learning rate | Best dev checkpoint | Dev acc (base 91.4) | Lost / gained questions vs base |
|---|---|---|---|---|
| Pilot | 1e-6 | step 20 | 90.2 | 14 / 8 |
| A | 1e-6 | step 32 | 91.2 | 10 / 9 |
| B | 3e-6 | step 32 | 92.2 → 91.2 at step 64 | 5 / 9, then 10 / 9 |
| C | 1e-5 | step 16 | 89.8 → 89.0 at step 32 | 17 / 9, then 18 / 6 |
| D (MATH + GSM8K) | 3e-6 | step 48 | 91.8 → 91.4 at step 64; **test 90.2** (base 90.4) | 9 / 11, then 8 / 8 |

- On GSM8K alone (A–C), the training reward stayed flat (about 0.70–0.80). The model was not learning to solve more training questions.
- On harder MATH problems (D), the reward rose, but the model mostly learned to write `#### N` instead of `\boxed{N}`. MATH strict went 72.6 → 82.2, but MATH accuracy stayed at 88.6 → 88.8, and GSM8K didn't move. Measuring both strict and lenient accuracy is what exposed this.
- Dev differences under about 1.5 points are noise: 45 of 500 dev questions switch between right and wrong from one model to the next.
- A higher learning rate made answers longer and more often cut off (2 → 15 at 8,192 tokens), not more accurate.
- **No model was released on Hugging Face**, because no checkpoint beat the base model. Full analysis and next ideas are in [EVAL_RESULTS.md §6.5–6.6](EVAL_RESULTS.md#65-grpo-results-no-run-beat-the-base-model). Total compute cost: about $66.

## How it works

```
GSM8K train ──► difficulty profiling ──► learnable questions ──► GRPO (TRL + vLLM) ──► eval (vLLM / MLX)
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
| `src/eval_vllm.py`, `scripts/eval_chain.sh` | Fast GPU evaluation (vLLM) with the same prompt and grader |
| `src/sweep.py`, `src/aggregate.py` | Run many eval settings from a YAML file, then summarize |
| `src/rewards.py` | RL reward functions (built on `grading.py`) |
| `src/profile_difficulty.py` | Samples 8 answers per train question and keeps the learnable ones |
| `src/make_math.py` | Builds MATH train/test sets with whole-number answers (run D) |
| `src/train_grpo.py` | GRPO training with TRL; `key=value` overrides on the command line |
| `configs/` | Eval sweeps (`phase1.yaml`, `qwen3_17b_baseline.yaml`) and training configs (`grpo_qwen3_17b.yaml`) |
| `scripts/runpod_setup.sh` | One-command setup on a cloud GPU pod |
| `tests/` | Reward function tests |
| `results/dev/summary.csv` | Summary of every eval run |
| `runs/configs/` | Config and `trainer_state.json` (per-step metrics) of every GRPO run |

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
python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=qwen3-17b-grpo-A max_steps=128 eval_n=null
python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=qwen3-17b-grpo-C learning_rate=1.0e-5 \
    max_steps=64 save_steps=16 eval_n=null max_completion_length=6144 vllm_max_model_length=6656 \
    per_device_train_batch_size=4 gradient_accumulation_steps=32 vllm_gpu_memory_utilization=0.3
python src/eval_vllm.py --model checkpoints/qwen3-17b-grpo-C/checkpoint-16 --thinking --tag think_vllm
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
- [x] Phase 2: Difficulty profiling, GRPO pilot, runs A / B / C, run D on harder MATH problems
- [x] Phase 3: Before/after analysis on dev: no gain over base, so no trained-model test run
- [ ] ~~Phase 4: Publish on Hugging Face~~ (skipped: nothing better than the base to publish)
- [ ] Future: 16 answers per question, length penalty, majority vote, GSM-Plus check

## Credits

- Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT)
- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0)
- Libraries: [MLX](https://github.com/ml-explore/mlx), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm)
