# Phase 1: Investigating the ≤6B GSM8K leaderboard

## The four models (as of 2026-11-23, all self-reported)
| Model | Params | Reported | Their setup | My dev score (same protocol) |
|---|---|---|---|---|
| ibm-granite/granite-4.1-3b | 3.4B | 86.88 | not stated | |
| microsoft/Phi-3.5-mini-instruct | 3.8B | 86.2 | Phi-3 tech report | |
| microsoft/Phi-3-mini-4k-instruct | 3.8B | 85.7 | 8-shot CoT | |
| Axonry/wsp81-qwen3.5-0.8b (was tobiasxxx/) | 0.87B | 84.03 | thinking on, T=0.6, 4 samples, LLM judge | |
| *(not run)* microsoft/Aion-1.0-Instruct-2.3B | 2.29B | 74.91 | added 2026-11-18; custom code, MLX unsupported | n/a |
| *(reference)* Qwen/Qwen3-0.6B | 0.6B | – | – | |

## Questions to answer
1. **Do the reported numbers reproduce?** Run all 4 under ONE fixed protocol on dev.
2. **How sensitive is each model to the protocol?** Change one thing at a time:
   - zero-shot vs 8-shot
   - strict extraction (`####` only) vs lenient (last number in the text)
   - Qwen3.5 only: thinking on vs off
3. **How do they fail?** (feeds Phase 2) Read 20+ wrong answers per model.
   Draft categories: arithmetic slip / misread question / wrong plan /
   right answer but unparseable / ran out of tokens.
4. **Cost vs accuracy:** tokens per answer (`n_tokens` in results). Is the 0.8B
   model "buying" its score with long thinking?

## Suggested order
1. Implement the 3 TODOs in src/eval.py. Test on `--n 20` with Qwen3-0.6B.
2. Read those 20 outputs by hand. Fix your extractor. (Expect to iterate!)
3. Run all 4 models on the full dev set (500) with your baseline protocol.
4. Run the protocol variations.
5. Write findings below.

## Decisions log
- 2026-11-23: dev = 500 from train (seed 42). Baseline decoding: greedy, max 1024 tokens.

- 2026-11-24: Pinned every model to an exact commit sha (configs/phase1.yaml). All are
  bf16 and unquantized, as published. The Qwen3.5 repo was renamed to Axonry/ and modified on 2026-11-24.

## Findings
- 2026-11-23: Qwen3-0.6B ignored the "####" instruction and used `\boxed{}`, so
  extraction rules matter.
