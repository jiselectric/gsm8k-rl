# gsm8k-rl

**A controlled evaluation of small models on GSM8K, followed by a pilot and four RLVR fine-tuning runs (GRPO) of Qwen3-1.7B on a RunPod H100.**

**Short answer:** before our GRPO runs, the original Qwen3-1.7B checkpoint (thinking on) scored **91.4 on the 500-question dev split** and **90.4 on the 1,319-question official test split**. None of the runs gave a confirmed accuracy gain. Run B's best checkpoint reached 92.2 on dev at step 32, then fell back to 91.2 at step 64. Run D's selected checkpoint scored **90.2 on test**: 1,190 correct, against 1,192 for the original checkpoint. So we did not release a model on Hugging Face.

**The main finding is about reward design.** In run D, the training reward rose, and the score that requires our answer format rose on MATH (72.6 → 82.2). But the format-tolerant score stayed flat (88.6 → 88.8, one more correct answer). The reward had mostly taught the answer format, not problem solving. Section 6.6 of [EVAL_RESULTS.md](EVAL_RESULTS.md#66-run-d-training-on-harder-math-problems) explains how.

What this repo shows:
- **An evaluation pipeline**: one prompt and one grader shared by evaluation and training, pinned model versions, and every answer saved per question.
- **A controlled comparison**: every checkpoint is compared with the original checkpoint on the same questions, settings and grader, question by question.
- **Two scores for every result**: one that requires the `#### N` format (strict) and one that accepts any answer format (acc). Without both, the run D format effect would have looked like a 10-point skill gain.

GSM8K is a set of grade-school math word problems. A model must solve each one and give a single number. Only the final number is graded.

## Evaluation results

One fixed setup for every model: zero-shot, greedy, one answer per question, the same answer-extraction rules. These are **dev** scores (500 questions held out from GSM8K *train*). Details are in [EVAL_RESULTS.md](EVAL_RESULTS.md).

| Model | Size | Best setting | Our dev score | Leaderboard claim (test, the team's own setup) |
|---|---|---|---|---|
| granite-4.1-3b | 3.4B | zero-shot | 94.2 | 86.88 |
| Phi-3-mini-4k | 3.8B | zero-shot | 89.6 | 85.7 |
| Phi-3.5-mini | 3.8B | zero-shot | 87.0 | 86.2 |
| **Qwen3-1.7B** (original checkpoint) | 1.7B | thinking | **91.4** (official test: **90.4**) | – |
| Qwen3-0.6B | 0.6B | thinking | 77.2 | – |
| Axonry Qwen3.5-0.8B | 0.87B | thinking | 76.8 | 84.03 |

Our dev scores and the leaderboard claims use different splits and different grading, so the columns can't be compared directly. Only Qwen3-1.7B was scored on the official test split. Some models may have trained on GSM8K train, which our dev split comes from.

Key findings:
1. **Leaderboard scores are not comparable.** They are self-reported, and each team uses its own split, prompt, sampling and grader.
2. **Few-shot examples hurt every model.** Models copy the short example style and reason less.
3. **Small models break on long problems**: 5+ step problems cost them 20–25 points.
4. **82% of training questions are "too easy"** for Qwen3-1.7B, so GRPO would learn nothing from them. We filter for questions the model *sometimes* gets right (1,029 of 6,973).

## GRPO results

All checkpoints were scored on dev with the same setup. The original checkpoint scores 91.4 on dev.

| Run | Learning rate | Training data | Dev acc of saved checkpoints | Lost / gained vs original checkpoint |
|---|---|---|---|---|
| Pilot | 1e-6 | GSM8K, learnable | 90.2 (step 20) | 14 / 8 |
| A | 1e-6 | GSM8K, learnable | 91.2 (step 32), 91.0 (step 64) | 10 / 9, 12 / 10 |
| B | 3e-6 | GSM8K, learnable | 92.2 (step 32), 91.2 (step 64) | 5 / 9, 10 / 9 |
| C | 1e-5 | GSM8K, learnable | 89.8 (step 16), 89.0 (step 32) | 17 / 9, 18 / 6 |
| D | 3e-6 | MATH + GSM8K, learnable | 89.4, 91.0, **91.8 (step 48)**, 91.4 | 9 / 11 at step 48 |

Final checks of run D step 48 (picked by its dev score), each run once:

| | Original checkpoint | Run D step 48 |
|---|---|---|
| GSM8K official test (1,319), acc | 90.4 (1,192) | 90.2 (1,190) |
| MATH test sample (500), strict: `#### N` only | 72.6 | 82.2 |
| MATH test sample (500), acc: any answer format | 88.6 | 88.8 |

- **No confirmed gain.** Run B's 92.2 didn't hold: the same run fell to 91.2. 71 of the 500 dev questions are right in some of our 12 GPU evaluations and wrong in others. So dev differences of a point or two are within noise.
- **On GSM8K data alone (A–C), the training reward showed no upward trend.** A higher learning rate (C) made answers longer and more often cut off (2 → 15 on dev), not more accurate.
- **On MATH (D), the reward rose because the model learned the format.** It switched from `\boxed{N}` to the `#### N` that the reward pays for. (`\boxed{N}` is LaTeX for "N in a box". MATH solutions mark their final answer that way, so models pick up the habit on MATH-style problems.) GSM8K answers already used `####`, so nothing carried over.
- **No model was released on Hugging Face**, because no checkpoint beat the original one. Total compute cost: about $66. Full analysis is in [EVAL_RESULTS.md §6.5–6.6](EVAL_RESULTS.md#65-grpo-results-no-run-beat-the-original-checkpoint).

## How it works

```
GSM8K / MATH train ──► difficulty profiling ──► learnable questions ──► GRPO (TRL + vLLM) ──► eval (vLLM / MLX)
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
| `configs/` | Eval sweeps (`phase1.yaml`, `qwen3_17b_baseline.yaml`) and training configs (`grpo_qwen3_17b.yaml` for the pilot and A–C, `grpo_qwen3_17b_math.yaml` for D) |
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
# run B (run C: learning_rate=1.0e-5 max_steps=64 save_steps=16)
python src/train_grpo.py configs/grpo_qwen3_17b.yaml run_name=qwen3-17b-grpo-B learning_rate=3.0e-6 \
    max_steps=96 save_steps=32 eval_n=null max_completion_length=6144 vllm_max_model_length=6656 \
    per_device_train_batch_size=4 gradient_accumulation_steps=32 vllm_gpu_memory_utilization=0.3
# run D: MATH + GSM8K
python src/make_math.py                                       # -> data/math_train.jsonl, data/math_test.jsonl
python src/profile_difficulty.py configs/grpo_qwen3_17b_math.yaml \
    --input data/math_train.jsonl --output data/math_filtered.jsonl --tag math
cat data/math_filtered.jsonl data/train_filtered.jsonl > data/train_mix.jsonl
python src/train_grpo.py configs/grpo_qwen3_17b_math.yaml
# score a checkpoint (dev by default; --split test or --split math_test for final checks)
python src/eval_vllm.py --model checkpoints/qwen3-17b-grpo-D/checkpoint-48 --thinking --tag think_vllm
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
- [x] Phase 3: Before/after analysis: every checkpoint on dev, then run D step 48 once on the official GSM8K test (90.2 vs 90.4) and on a 500-problem MATH test sample. No confirmed gain.
- [ ] ~~Phase 4: Publish on Hugging Face~~ (skipped: no checkpoint beat the original checkpoint)
- [ ] Future: accept `\boxed{N}` in the reward when training on MATH, 16 answers per question, length penalty, test-split check of the other models, GSM-Plus check. Majority vote too, reported separately because it changes the scoring, not the model.

## Credits

- Datasets: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT), [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (MATH, MIT)
- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0)
- Libraries: [MLX](https://github.com/ml-explore/mlx), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm)
