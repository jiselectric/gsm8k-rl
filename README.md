# gsm8k-rl

Investigating why small LLMs succeed or fail on GSM8K, then training a small Qwen model
with reinforcement learning from verifiable rewards (GRPO) to beat them.

## Setup
```bash
conda activate gsm8k          # or: conda env create -f environment.yml
python src/make_splits.py     # already done; creates data/{train,dev,test}.jsonl
python src/hello.py           # smoke test: one question, Qwen3-0.6B
```

## Layout
| Path | What |
|---|---|
| `data/` | `train` (6,973) / `dev` (500, held out from train) / `test` (1,319) |
| `src/make_splits.py` | builds the fixed splits (seed 42) |
| `src/hello.py` | minimal load-and-generate example |
| `src/eval.py` | eval harness: **3 TODOs to implement** |
| `results/<split>/` | one JSONL per (model, config) run, with every completion |
| `notebooks/` | analysis (use kernel "Python (gsm8k)") |
| `notes/` | lab notebook: decisions, findings, open questions |

## Rules
1. Develop on `dev`. Touch `test` only for final reported numbers.
2. Every reported number must record its model, prompt, decoding settings, and extraction rule.

## Roadmap
- [ ] Phase 1: Re-evaluate the 4 leaderboard models (≤6B) under one protocol
- [ ] Phase 2: Error analysis: categorize the failures
- [ ] Phase 3: GRPO training on a small Qwen (cloud GPU)
- [ ] Phase 4: Before/after analysis, HF submission, write-up
