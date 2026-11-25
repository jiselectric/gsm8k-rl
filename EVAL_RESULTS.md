# GSM8K evaluation results and analysis

All evaluation results for this project in one place: what we measured, how, what we found, and what it means for training.

*Last updated: 2026-11-25. Two runs are still in progress (see [Status](#status)).*

---

## 1. Summary

- We re-tested the top **≤6B models on the GSM8K leaderboard** with **one fixed, strict test setup**, plus two small Qwen models.
- **Leaderboard numbers are not comparable.** Every team scores its own model with its own setup. Under our setup, some models scored *higher* than their claim, and one scored *lower*.
- **Solved examples in the prompt (few-shot) made every model worse.** The models copied the short example style and reasoned less.
- **Thinking mode helps small Qwen models a lot** (+17 to +19 points), but costs 5–10x more tokens.
- **Small models fail on long problems.** The gap between a 0.6B and a 3B model is small on 2-step problems and large on 5+ step problems.
- **Qwen3-1.7B is the training base.** Without thinking it already scores 84.0. With thinking, it solves most training questions every time, so we **filter training data by difficulty** before RL.

---

## 2. How we tested (the protocol)

Every number below uses the same setup unless the table says otherwise.

| Setting | Value |
|---|---|
| Data | **dev** split: 500 questions held out from GSM8K *train* (seed 42) |
| Prompt | One system prompt for all models: *"…Think step by step… End your response with the final numeric answer in the form '#### &lt;number&gt;'."* |
| Shots | 0 (zero-shot) unless noted. 4-shot and 8-shot examples come from train, with calculator notes `<<…>>` removed |
| Decoding | Greedy (temperature 0), max 1,024 tokens (8,192 for thinking mode), batch size 8 |
| Model versions | Every model is pinned to an exact Hugging Face commit (`configs/phase1.yaml`) |
| Hardware | Apple M3 Pro (MLX), bf16 weights as published |

### How answers are graded

GSM8K only grades the **final number**. The reasoning steps are never checked.

1. Find the answer in the model's text, using these rules in order:
   `#### N` → `\boxed{N}` → "the answer is N" → the last number in the text.
2. Compare it to the gold number by **value**: `72`, `72.0` and `72.00` are all equal, and `1,250` equals `1250`.

We report two scores:

| Score | Meaning |
|---|---|
| **acc** (lenient) | Correct under *any* rule above |
| **strict** | Correct *and* written as `#### N`, the format we asked for |

The gap between the two shows how well a model follows the format instruction.

Uncertainty: with 500 questions, each score is about **±2–4 points** (95% confidence interval).

---

## 3. The models

| Model | Size | Leaderboard claim (test) | How they measured it |
|---|---|---|---|
| ibm-granite/granite-4.1-3b | 3.4B | 86.88 | Not stated |
| microsoft/Phi-3.5-mini-instruct | 3.8B | 86.2 | Phi-3 tech report |
| microsoft/Phi-3-mini-4k-instruct | 3.8B | 85.7 | 8-shot chain-of-thought |
| Axonry/wsp81-qwen3.5-0.8b | 0.87B | 84.03 | Thinking on, 4 samples at temperature 0.6, an LLM judges correctness |
| *Not run:* microsoft/Aion-1.0-Instruct-2.3B | 2.3B | 74.91 | Needs custom model code that MLX can't run |
| *Reference:* Qwen/Qwen3-0.6B | 0.6B | – | – |
| *Training base:* Qwen/Qwen3-1.7B | 1.7B | – | – |

The leaderboard is **self-reported**: each model owner uploads their own score. Hugging Face does not run the evaluation.

---

## 4. Full results (dev, 500 questions)

| Model | Setting | acc | strict | Uses `####` | Cut off | Avg tokens | Sec / question |
|---|---|---|---|---|---|---|---|
| granite-4.1-3b | **base** | **94.2** | **94.2** | 100% | 0% | 113 | 1.4 |
| granite-4.1-3b | 4-shot | 91.6 | 91.6 | 100% | 0% | 74 | 1.6 |
| granite-4.1-3b | 8-shot | 91.8 | 91.8 | 100% | 0% | 72 | 2.3 |
| Phi-3-mini-4k | **base** | **89.6** | 17.8 | 19% | 0% | 178 | 2.6 |
| Phi-3-mini-4k | 4-shot | 87.2 | 87.0 | 99% | 0% | 92 | 2.3 |
| Phi-3-mini-4k | 8-shot | 84.0 | 84.0 | 100% | 0% | 80 | 3.0 |
| Phi-3.5-mini | **base** | **87.0** | 80.6 | 92% | 0% | 186 | 2.5 |
| Phi-3.5-mini | 4-shot | 79.6 | 69.4 | 86% | 0% | 138 | 3.2 |
| Phi-3.5-mini | 8-shot | 78.6 | 70.4 | 87% | 0.2% | 131 | 4.1 |
| **Qwen3-1.7B** | **base** | **84.0** | **82.8** | 98% | 0.4% | 207 | 1.5 |
| Qwen3-1.7B | think | *running* | | | | | |
| Qwen3-0.6B | base | 60.6 | 37.2 | 62% | 0% | 182 | 0.5 |
| Qwen3-0.6B | 4-shot | 48.8 | 48.4 | 97% | 0.2% | 117 | 0.5 |
| Qwen3-0.6B | 8-shot | 47.8 | 47.4 | 98% | 0% | 106 | 0.7 |
| Qwen3-0.6B | **think** | **77.2** | 15.6 | 21% | 4.8% | 1,125 | 7.1 |
| Axonry Qwen3.5-0.8B | base | 57.6 | 57.4 | 89% | 10.2% | 423 | 1.5 |
| Axonry Qwen3.5-0.8B | 4-shot | 56.2 | 56.2 | 95% | 4.8% | 249 | 1.3 |
| Axonry Qwen3.5-0.8B | 8-shot | 53.2 | 53.0 | 96% | 4.2% | 181 | 1.2 |
| Axonry Qwen3.5-0.8B | **think** | **76.8** | 74.4 | 90% | 9.4% | 3,954 | 23.3 |

Bold marks each model's best setting. "Cut off" means the answer hit the token limit before finishing. Raw numbers are in `results/dev/summary.csv`.

---

## 5. Findings

### 5.1 Leaderboard numbers don't reproduce, in both directions

| Model | Claim (test) | Ours (dev, best setting) | Gap |
|---|---|---|---|
| granite-4.1-3b | 86.88 | 94.2 | **+7.3** |
| Phi-3-mini-4k | 85.7 | 89.6 | +3.9 |
| Phi-3.5-mini | 86.2 | 87.0 | +0.8 |
| Axonry Qwen3.5-0.8B | 84.03 | 76.8 | **−7.2** |

Why the gaps exist:

- **Our dev set comes from GSM8K *train*.** Many models were fine-tuned on GSM8K train, so they may have seen these exact questions. This can inflate dev scores for big models.
  - We checked for copying: models' answers share only about **12%** of 5-word phrases with the reference solutions. They are not reciting memorized text, but they may still be "familiar" with the questions.
  - The fair check is to run the same models on the **test** split. *(Not done yet.)*
- **Axonry's claim used a much looser setup.** It took 4 samples, allowed 32k tokens, and had an LLM judge answers. We use one greedy answer, an 8k limit, and exact matching. 9.4% of its answers ran out of tokens under our limit.
- **Different graders give different scores.** See 5.4.

**Lesson:** a leaderboard score only means something together with its exact test setup.

### 5.2 Few-shot examples made every model worse

| Model | Questions lost (0→4-shot) | Questions gained | Net |
|---|---|---|---|
| Qwen3-0.6B | 98 | 39 | −59 |
| Phi-3.5-mini | 62 | 25 | −37 |
| Phi-3-mini-4k | 32 | 20 | −12 |
| granite-4.1-3b | 22 | 9 | −13 |
| Axonry Qwen3.5-0.8B | 75 | 68 | −7 |

**Why:** the GSM8K example solutions are short, calculator-style lines. The models copy that style and write shorter answers:

```
avg tokens per answer     0-shot   4-shot   8-shot
Qwen3-0.6B                  182      117      106
Phi-3.5-mini                186      138      131
granite-4.1-3b              113       74       73
```

For a language model, writing steps *is* thinking. Shorter answers mean less thinking and more mistakes. One example of a Phi-3.5 4-shot error:

```
Ben has 40+8=48 apples.
Tom has 48*(3/8)=36 apples.
#### 36          (correct answer: 18; a step was skipped)
```

Few-shot was popular with older **base** models (like GPT-3), which didn't know the task. All models here are **instruction-tuned**: they already know the task, so examples act only as a style guide, and here a bad one. The DeepSeek-R1 paper (2025) reported the same: few-shot prompts hurt reasoning models.

**The one thing few-shot fixed is formatting.** Phi-3-mini's strict score jumped from 17.8 to 87.0 at 4-shot, because it finally used `####`.

### 5.3 Thinking mode helps small models, at a high token cost

| Model | Base | Think | Gain | Token cost |
|---|---|---|---|---|
| Qwen3-0.6B | 60.6 | 77.2 | +16.6 | 6x more tokens |
| Axonry Qwen3.5-0.8B | 57.6 | 76.8 | +19.2 | 9x more tokens |

Qwen3-0.6B gets the same score as Axonry's model with **3.5x fewer tokens** (1,125 vs 3,954).

### 5.4 Following the format is a separate skill

| Model | Uses `####` (base) | acc | strict |
|---|---|---|---|
| granite-4.1-3b | 100% | 94.2 | 94.2 |
| Qwen3-1.7B | 98% | 84.0 | 82.8 |
| Phi-3-mini-4k | **19%** | 89.6 | **17.8** |
| Qwen3-0.6B (think) | **21%** | 77.2 | **15.6** |

Phi-3-mini is good at math, but it mostly ignores the `####` instruction. A strict grader would score it 17.8 and a lenient grader 89.6. **Same model, same answers, 72-point difference.**

#### The `82.0` vs `82` problem

All GSM8K gold answers are whole numbers. Some models write `#### 82.0`. Our grader compares values, so it counts this as correct. The most common tool, lm-evaluation-harness, compares **text** after removing `,`, `$` and a trailing `.`, so it marks `82.0` **wrong**.

Correct answers that only pass because of number cleanup:

```
run                     decimal (82.0)   comma (1,000)
Phi-3-mini base               11               8
Qwen3-0.6B base               11               0
Qwen3-0.6B think              11               2
Phi-3.5-mini base              6               3
Axonry think                   1               1
granite (all runs)             0               0
```

That is about **2% of questions** for some models, enough to move a model up or down a close leaderboard. Our training reward gives `#### 82.0` only **half** credit, so the trained model learns to write clean whole numbers that every grader accepts.

### 5.5 Small models fail on long problems

Accuracy by the number of calculation steps in the reference solution (best setting per model):

```
steps   questions   granite   Phi-3   Phi-3.5   Qwen3-1.7B   Qwen3-0.6B   Axonry
                    3B        3.8B    3.8B      (base)       (think)      (think)
1-2        188        96        94       91         91           85          83
3          148        94        92       90         87           78          82
4           92        96        84       87         77           68          68
5+          72        88        82       71         67           65          60
```

- On short problems, all models are close (83–96).
- On 5+ step problems, small models drop by 20–25 points, while Granite stays near 90.

**This is the real weakness RL must fix:** staying correct over many steps. Fixing the format alone won't close the gap.

### 5.6 Hard questions and possibly bad labels

Only **11 of 500** questions are missed by every model's best setting. If we could always pick the best model per question, accuracy would be 97.8%. Some of those 11 may have wrong gold answers, which is a known GSM8K issue. They still need checking by hand.

---

## 6. From evaluation to training

### 6.1 Choosing the base model

| Option | Pros | Cons |
|---|---|---|
| Qwen3-0.6B | Smallest, cheap RL | Starts at 77 (think); 88 is unlikely |
| Qwen3.5-0.8B | Newest | Unused vision part (101M params), slow attention without special GPU code, very long thinking |
| **Qwen3-1.7B** ✅ | Already 84.0 without thinking, follows the format | ~2–3x more expensive RL than 0.6B |

We chose **Qwen3-1.7B with thinking on**, because it has the best chance of reaching the top of the ≤6B leaderboard (88+).

### 6.2 Difficulty profiling: most questions are too easy

GRPO (the RL method) learns only by **comparing several answers to the same question**:

```
8 answers: ✓✓✓✓✓✓✓✓  → all the same → no learning signal
8 answers: ✗✗✗✗✗✗✗✗  → all the same → no learning signal
8 answers: ✓✓✗✓✗✓✓✗  → mixed        → the model learns from this
```

So before training, we sample 8 answers per training question, set up exactly like training: thinking on, temperature 1.0, a 4,096-token limit, and the strict reward. First test, 64 training questions:

```
right out of 8   questions
  0/8               3  ##
  1–7/8            13  ######                          ← learnable (20%)
  8/8              48  ##############################
```

| Measure | Value |
|---|---|
| Accuracy across all 512 sampled answers | 87.7% (strict = lenient: format is perfect) |
| Always right / learnable / always wrong | 75% / 20% / 5% |
| Answers cut off at 4,096 tokens | 2.1% |
| Avg answer length | 756 tokens |
| Speed (1x H100, vLLM) | 512 answers in 23 s (~16k tokens/s) |

Without filtering, **about 75% of training compute would teach the model nothing.** Training uses only the learnable questions (`data/train_filtered.jsonl`). *(The full profile of all 6,973 questions is running.)*

### 6.3 Training plan

| Run | What changes | Why |
|---|---|---|
| Pilot | 20 steps | Check that the reward rises, memory is OK, and few groups lack signal |
| A | Plain GRPO on filtered data, 200 steps | First trained model |
| B | A + clip-higher (`epsilon_high=0.28`) + no reward scaling (`scale_rewards=none`) | Measures whether these published tricks (DAPO, Dr. GRPO) help |

Every 50 steps we also check the reward on 200 dev questions, with no learning from them. If the training reward rises but the dev reward doesn't, the model is overfitting.

**Things we chose not to do:**
- **SFT on reference solutions.** Section 5.2 shows short human solutions make models reason less.
- **Learned reward models.** The answer can be checked exactly, so there's no need for one.
- **Training on GSM-Plus / GSM-Symbolic.** These stay untouched, to test whether the model learned to reason rather than memorize GSM8K.

---

## Status

| Item | Status |
|---|---|
| Phase-1 sweep (6 models × settings) | ✅ Done |
| Qwen3-1.7B base | ✅ 84.0 |
| Qwen3-1.7B think | ⏳ Running (Mac) |
| Difficulty profiling, full train set | ⏳ Running (H100) |
| Test-split check for Granite / Phi-3 (contamination) | ☐ To do |
| Stock Qwen3.5-0.8B runs | ☐ Skipped (base model changed to Qwen3-1.7B) |
| GRPO pilot, run A, run B | ☐ Next |
| Final test-set number + GSM-Plus generalization | ☐ After training |

## Decisions log

- **2026-11-23**: dev = 500 questions held out from train (seed 42). Baseline: greedy, max 1,024 tokens.
- **2026-11-23**: Qwen3-0.6B ignored `####` and used `\boxed{}`, so we added several answer-finding rules (Section 2).
- **2026-11-24**: Pinned every model to an exact commit. The Qwen3.5 model repo was renamed from `tobiasxxx/` to `Axonry/` (same files).
- **2026-11-24**: Switched to batched generation (batch 8, ~3.7x faster). Batched outputs differ very slightly from unbatched ones, so we only compare runs with the same batch size.
- **2026-11-24**: Chose Qwen3-1.7B (thinking) as the RL base model.
- **2026-11-25**: Added difficulty filtering before RL, after the 64-question test showed 75% always-right questions.

## Reproduce

```bash
python src/sweep.py configs/phase1.yaml            # all phase-1 runs (resumes, skips finished)
python src/sweep.py configs/qwen3_17b_baseline.yaml
python src/aggregate.py                            # -> results/dev/summary.csv
```
