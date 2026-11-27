# GSM8K evaluation results and analysis

All evaluation results for this project in one place: what we measured, how, what we found, and what it means for training.

*Last updated: 2026-11-26. A pilot and four GRPO runs finished. None gave a confirmed GSM8K gain over the original checkpoint (see [6.5](#65-grpo-results-no-run-beat-the-original-checkpoint) and [6.6](#66-run-d-training-on-harder-math-problems)).*

**Words used here.** *Original checkpoint* means Qwen/Qwen3-1.7B at commit `70d244c`, with thinking on, before our GRPO runs. *Dev* is our 500-question split held out from GSM8K train. *Test* is the official 1,319-question GSM8K test split.

---

## 1. Summary

- We re-tested the top **≤6B models on the GSM8K leaderboard** with **one fixed evaluation setup** on our dev split, plus two small Qwen models.
- **Leaderboard numbers are not comparable.** Every team scores its own model with its own setup. Under our setup (on dev, not test), some models scored *higher* than their claim, and one scored *lower*.
- **Solved examples in the prompt (few-shot) made every model worse.** The models copied the short example style and reasoned less.
- **Thinking mode helps small Qwen models a lot** (+17 to +19 points), but costs 5–10x more tokens.
- **Small models fail on long problems.** The gap between a 0.6B and a 3B model is small on 2-step problems and large on 5+ step problems.
- **Qwen3-1.7B is the training base.** With thinking on, the original checkpoint scores **91.4 on dev** and **90.4 on the official test split** (1,192 of 1,319; strict 90.2). That is numerically above every ≤6B leaderboard claim, but those claims are self-reported with other protocols, so it is not a like-for-like ranking. The original checkpoint solves 82% of training questions in all 8 sampled attempts, so we **filter training data by difficulty** before RL.
- **GSM8K-only GRPO (pilot, A, B, C) gave no confirmed gain.** Run B reached 92.2 on dev at step 32, then fell to 91.2 at step 64. The highest learning rate made it worse (89.0 vs 91.4), mainly by making answers run past the length limit.
- **Run D (harder MATH problems) raised the training reward, but through format, not skill.** On a 500-problem MATH test sample, the strict score (requires `#### N`) rose from 72.6 to 82.2. The format-tolerant score went from 88.6 to 88.8, one more correct answer. On the official GSM8K test, run D's selected checkpoint scored 90.2 (1,190 correct vs 1,192). Section 6.6 explains why these results fit together.
- **Conclusion:** a 1.7B model that has already had RL and sits near 91% on GSM8K has little left that this budget of RL can teach it. The most useful result is how a correctness-plus-format reward can rise without any gain in problem solving.

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
| **Qwen3-1.7B** | **think** | **90.4** | **90.2** | 99% | 0.4% | 799 | 9.9 |
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

Only **11 of 500** questions are missed by every model's best setting. If we could always pick the best model per question, accuracy would be 97.8%. Some of those 11 may have wrong gold answers, which is a known GSM8K issue. Section 6.5 has a related count for Qwen3-1.7B: 23 dev questions are wrong for the original checkpoint and for every trained checkpoint.

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

**Full profile, all 6,973 training questions (55,784 sampled answers):**

```
right out of 8   questions
  0/8              234   (3.4%)  always wrong
  1/8               77  ┐
  2/8               75  │
  3/8               63  │
  4/8               82  ├ 1,029 LEARNABLE (15%)
  5/8               98  │
  6/8              155  │
  7/8              479  ┘
  8/8            5,710   (82%)   always right
```

| Measure | Value |
|---|---|
| Accuracy across all sampled answers | 91.8% strict, 91.9% lenient |
| Cut off at 4,096 tokens | 2.0% |
| Avg answer length | 755 overall; **1,522 on learnable questions**; 1,940 on always-wrong ones |
| Cost | ~35 min on 1x H100 (~$2.50) |

- Without filtering, **82% of training compute would teach the model nothing.** Training uses only the 1,029 learnable questions (`data/train_filtered.jsonl`).
- Harder questions get longer answers (2x on learnable questions).
- None of the always-wrong questions failed on format alone. The math was wrong.

### 6.3 Training plan

*This is the plan as written before the pilot. The pilot changed it: run A was cut to 128 planned steps, and run B dropped clip-higher (it can't have an effect here, see 6.4). What actually ran is in 6.5 and 6.6.*

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

### 6.4 First GPU results: the original checkpoint is already strong

All evaluated with vLLM on the H100: greedy, thinking on, 8,192-token limit, same prompt and grader.

| Model | Split | acc | strict | Avg tokens | Cut off |
|---|---|---|---|---|---|
| Original checkpoint | dev (500) | 91.4 | 91.4 | 816 | 2 |
| **Original checkpoint** | **test (1,319)** | **90.4** (1,192) | **90.2** (1,190) | 959 | 15 |
| Pilot (20 GRPO steps) | dev (500) | 90.2 | 90.2 | 825 | 3 |

- **Before our GRPO runs, the original checkpoint scores 90.4 on the official test split.** That is numerically above every ≤6B leaderboard claim (the highest is Granite's 86.88). But the claims are self-reported under each team's own protocol, and we did not score Granite or Phi on test (Granite scored 94.2 on our *dev* split). So this shows a strong 1.7B model under a strict setup, not a verified leaderboard rank.
- **MLX (Mac) and vLLM (GPU) agree.** The original checkpoint is right or wrong on the same question in 481 of 500 dev questions (90.4 vs 91.4).
- **Dev ≈ test for Qwen** (91.4 vs 90.4), so Qwen's dev score does not look inflated. We have not run this check for the other models (see 5.1).
- **The pilot did not help** (lost 14, gained 8 questions vs the original checkpoint). Its logs show why: answers grew longer, and the share cut off at 4,096 tokens rose from 3.5% to about 11%. We had set cut-off answers to be *ignored* in the loss, so rambling cost nothing. **Fix for run A: cut-off answers count as wrong (reward 0).**
- Pilot logs also showed `clip_ratio = 0` on every step. With one update per batch, the policy ratio is always 1, so "clip-higher" (`epsilon_high`) **cannot have any effect** in this setup. Run B is redesigned around settings that matter here: more answers per question, or no reward scaling.

**What RL must do now:** improve a model that is already at 90%. Realistic goal: **+1 to +2 points** and/or shorter answers. With 500 dev questions (±2.6 points), we compare checkpoints question by question, not only by raw accuracy.

### 6.5 GRPO results: no run beat the original checkpoint

All runs: Qwen3-1.7B, thinking on, 1,029 filtered "learnable" questions, 8 answers per question, 16 questions per step, temperature 1.0, no KL penalty, cut-off answers get reward 0. Dev scores use vLLM, greedy decoding, an 8,192-token limit, and the same grader as everything else.

| Run | Learning rate | Answer limit | Steps done | Checkpoint | Dev acc | Lost vs original | Gained vs original | Avg tokens | Cut off |
|---|---|---|---|---|---|---|---|---|---|
| Original checkpoint | - | - | - | - | **91.4** | - | - | 816 | 2 |
| Pilot | 1e-6 | 4,096 | 20 | final | 90.2 | 14 | 8 | 824 | 3 |
| A | 1e-6 | 4,096 | 87 / 128 | step 32 | 91.2 | 10 | 9 | 839 | 5 |
| A | | | | step 64 | 91.0 | 12 | 10 | 811 | 3 |
| B | 3e-6 | 6,144 | 66 / 96 | step 32 | 92.2 | 5 | 9 | 789 | 3 |
| B | | | | step 64 | 91.2 | 10 | 9 | 845 | 4 |
| C | 1e-5 | 6,144 | 42 / 64 | step 16 | 89.8 | 17 | 9 | 820 | 14 |
| C | | | | step 32 | 89.0 | 18 | 6 | 855 | 15 |

**What we saw:**

- **The training reward showed no upward trend.** Single steps ranged from about 0.40 to 0.95, but the 8-step average stayed around 0.70–0.80 for the whole of every run.
  - The first steps were already above the profiling estimate of 0.67. That is expected: questions were kept *because* one 8-answer draw was imperfect, so a fresh draw of the same question often scores higher (regression to the mean).
  - The same effect weakens the signal. On average **39–43% of groups** in each run had all 8 answers with the same reward (`frac_reward_zero_std` in `runs/configs/*/trainer_state.json`), so they taught nothing.
- **B step 32 (92.2) did not hold.** It gained 9 questions and lost 5, but thirty-two steps later the same run was back to 91.2. Across our 12 GPU evaluations on dev (the original checkpoint plus 11 trained checkpoints), **71 of the 500 questions** are right in some evaluations and wrong in others. So dev differences of a point or two are within noise.
- **A higher learning rate changed the model, but for the worse.** Run C (1e-5) lost 18 questions and gained 6. The number of answers cut off at 8,192 tokens jumped from 2 to 15. The model learned to think longer without thinking better.
- **23 dev questions are wrong in all 12 of those evaluations.** Several look like unclear wording or a questionable label (Section 5.6), so about 95 is a practical ceiling on this dev set.

**Why RL didn't help here (our best explanation):**

1. **Qwen3-1.7B is already RL-trained.** Qwen trained its thinking mode with large-scale RL on math, so GSM8K is already in its comfort zone. Most of the 1,029 "learnable" questions are solved 5–7 times out of 8, so a group of 8 answers often has only 1–2 wrong ones, and the learning signal is weak and noisy.
2. **The remaining mistakes are not of the kind RL fixes easily.** They are misreadings of the question, unusual wording, and questionable labels, rather than a reasoning skill the model could practice.
3. **The budget was small.** About 200 total steps across the runs (~3,200 questions × 8 answers). Published gains usually come from thousands of steps on harder, more varied math data.

**What would be worth trying next:**

- Train on **harder math** (e.g. MATH, NuminaMath) instead of GSM8K's own training set, then test whether GSM8K improves as a side effect. *Done as run D (Section 6.6).*
- **16 answers per question**, so groups that are almost always right still contain some wrong answers to learn from.
- A **length penalty** or a shorter answer limit, to stop the "think longer" drift seen in run C.
- **Majority vote at test time** (sampling several answers and picking the most common). This often adds 1–3 points on GSM8K, but it changes how the model is scored rather than the model itself, so it doesn't count as a leaderboard result for the model.

**Decision:** we did **not** publish a trained model to Hugging Face. No checkpoint was better than the original checkpoint, so releasing one would mislead people. The original checkpoint's 90.4 on test (Section 6.4) is the honest headline number.

**Cost:** about $34 of RunPod credit (H100 SXM at $4.39/hr, plus storage). The pod and the network volume were deleted when training ended.

### 6.6 Run D: training on harder math problems

Runs A–C had a weak learning signal: most "learnable" GSM8K questions were already right 6–7 times out of 8. Run D trains on **harder problems** instead.

**Data.** We kept the [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) *train* problems that have a whole-number answer (4,786, so the same `#### N` grader works). We profiled them like GSM8K (8 answers each, 6,144-token limit for MATH). Outcomes below are what we observed under the **format-sensitive (strict) grader**, the same rule as the training reward. Source: `runs/difficulty_math.jsonl` and `runs/difficulty_qwen3-17b-grpo-v1.jsonl`.

| Strict passes out of 8 | GSM8K train (6,973) | MATH train (4,786) |
|---|---|---|
| 8 of 8 (no signal) | 5,710 (82%) | 1,952 (41%) |
| **1–7 of 8 (learnable)** | **1,029 (15%)** | **2,043 (43%)** |
| 0 of 8 (no signal) | 234 (3%) | 791 (17%) |
| Average answer length | 755 tokens | 2,549 tokens |
| Answers cut off | 2.0% | 17.5% |

The same MATH samples look very different under the **format-tolerant grader**: 3,158 problems (66%) are right in all 8 attempts and 505 (11%) in none. Much of what the strict grader marked as "learnable" or "always wrong" on MATH was a right answer in `\boxed{}` instead of `####`. On GSM8K the two graders almost agree (5,731 all-right, 234 none), because the model already uses `####` there.

Training set: the 2,043 learnable MATH problems plus the 1,029 learnable GSM8K questions. Settings: learning rate 3e-6, 6,144-token answers, 16 questions × 8 answers per step. Stopped at step 69 of 96.

**Training reward went up for the first time:** 0.66 (steps 1–8) → about 0.75–0.78 (steps 40–69). Runs A–C never moved.

**GSM8K dev (500):**

| Checkpoint | Dev acc | Lost / gained vs original | Avg tokens |
|---|---|---|---|
| Original checkpoint | 91.4 | – | 816 |
| D step 16 | 89.4 | 16 / 6 | 810 |
| D step 32 | 91.0 | 10 / 8 | 805 |
| D step 48 | 91.8 | 9 / 11 | 786 |
| D step 64 | 91.4 | 8 / 8 | 795 |

**Final checks** (step 48 was chosen because it had the best dev score; each test was run once):

| Model | GSM8K test (1,319): acc | MATH test sample (500, integer answers, never trained on): acc, any format | MATH test sample: strict, `#### N` only |
|---|---|---|---|
| Original checkpoint | **90.4** (1,192) | 88.6 (443) | 72.6 (363) |
| D step 48 | 90.2 (1,190; lost 34, gained 32) | 88.8 (444; lost 22, gained 23) | **82.2** (411) |
| D step 64 | – | 88.2 (441) | **82.2** (411) |

On GSM8K test, the strict scores are the same: 1,190 for both.

#### Why the reward rose but problem solving didn't

The training reward and the strict score measure the same thing: *right number* **and** *written as `#### N`*. Either part can move that score. The format-tolerant score (`acc`) measures only the first part. The evidence:

1. **The original checkpoint was often right in the wrong format on MATH.** On the MATH test sample it wrote `\boxed{N}` for 88 of 500 answers, the style MATH solutions use. A right answer in `\boxed{}` gets reward 0.
2. **GRPO rewards exactly that difference.** In one group of 8 answers, some say `\boxed{42}` and some say `#### 42`, with the same math. Only the `####` ones get reward 1, so the advantage points straight at the format. It is one habit that pays off on every problem. Getting a hard problem right more often needs changes specific to that problem, and the reward pays off less reliably.
3. **The training logs show it.** We log both rewards but train only on strict (`runs/configs/qwen3-17b-grpo-D/checkpoint-64_trainer_state.json`, averaged over steps):

   | Run D steps | Strict reward (trained on) | Format-tolerant reward (logged only) | Gap |
   |---|---|---|---|
   | 1–8 | 0.66 | 0.81 | 0.15 |
   | 9–24 | 0.71 | 0.83 | 0.12 |
   | 25–40 | 0.76 | 0.86 | 0.10 |
   | 41–56 | 0.77 | 0.85 | 0.08 |

   Strict rose by 0.11 but format-tolerant by only 0.04. Most of the rise came from closing the gap.
4. **The test results show it too.** On the MATH test sample, `\boxed{}` answers dropped from 88 to 34, and strict-correct answers rose from 363 to 411 (+48). Format-tolerant correct answers went from 443 to 444 (lost 22, gained 23).
5. **GSM8K had no format to fix.** The original checkpoint already ends about 99% of GSM8K answers with `####` (1,300 of 1,319 on test). So the one thing run D learned had nothing to improve on GSM8K, and the test change (lost 34, gained 32) is noise.

**Other effect:** answers got a little shorter (GSM8K test: 959 → 901 tokens; MATH: 2,474 → 2,343), with no loss in accuracy. That is a small efficiency gain, not a score gain.

**Lesson for reward design:** a reward that checks correctness *and* format will push on whichever part is cheaper to fix. Two fixes are to accept `\boxed{N}` in the reward when training on MATH, or to always report a format-tolerant score next to the strict one. We did the second. Without it, run D would have looked like a +9.6-point MATH gain.

**Cost:** about $32 for run D (profiling ~2h20m, training ~3h, checks ~45m). Total for the project: about $66 of RunPod credit.

## Status

| Item | Status |
|---|---|
| Phase-1 sweep (6 models × settings) | ✅ Done |
| Qwen3-1.7B base | ✅ 84.0 |
| Qwen3-1.7B think (original checkpoint) | ✅ 90.4 dev (MLX), 91.4 dev / **90.4 test** (vLLM) |
| Difficulty profiling, full train set | ✅ 1,029 learnable questions |
| GRPO pilot (20 steps) | ✅ Works; found length creep, fixed |
| Test-split check for Granite / Phi-3 (contamination) | ☐ To do |
| Stock Qwen3.5-0.8B runs | ☐ Skipped (base model changed to Qwen3-1.7B) |
| GRPO runs A / B / C (lr 1e-6 / 3e-6 / 1e-5) | ✅ Done, no confirmed gain over the original checkpoint (Section 6.5) |
| GRPO run D (MATH + GSM8K) | ✅ Done: format learned, no accuracy gain (Section 6.6) |
| Hugging Face release | ✖ Not done: no checkpoint beat the original checkpoint |
| Final test-set number | ✅ Original checkpoint 90.4 (1,192), run D step 48: 90.2 (1,190) |
| Accept `\boxed{N}` in the reward for MATH, 16 answers per question, length penalty, majority vote | ☐ Ideas for future work |

## Decisions log

- **2026-11-23**: dev = 500 questions held out from train (seed 42). Baseline: greedy, max 1,024 tokens.
- **2026-11-23**: Qwen3-0.6B ignored `####` and used `\boxed{}`, so we added several answer-finding rules (Section 2).
- **2026-11-24**: Pinned every model to an exact commit. The Qwen3.5 model repo was renamed from `tobiasxxx/` to `Axonry/` (same files).
- **2026-11-24**: Switched to batched generation (batch 8, ~3.7x faster). Batched outputs differ very slightly from unbatched ones, so we only compare runs with the same batch size.
- **2026-11-24**: Chose Qwen3-1.7B (thinking) as the RL base model.
- **2026-11-25**: Added difficulty filtering before RL, after the 64-question test showed 75% always-right questions.
- **2026-11-25**: Measured the original checkpoint once on test (90.4) as the official "before" number.
- **2026-11-25**: After the pilot, cut-off answers count as wrong (`mask_truncated_completions: false`). Run A = 128 steps (~2 passes over the learnable set).
- **2026-11-25**: Run A stopped at step 87 (flat reward, dev 91.0–91.2). Run B: learning rate 3e-6, 6,144-token answer limit, no in-training eval (it took 12 minutes each time).
- **2026-11-25**: Run B stopped at step 66 (step 32 got 92.2, but step 64 fell back to 91.2). Run C: learning rate 1e-5.
- **2026-11-25**: Run C stopped at step 42 (dev 89.8, then 89.0, with more cut-off answers). Stopped training and deleted the pod and volume. No Hugging Face release.
- **2026-11-26**: Run D: harder MATH problems (whole-number answers) + learnable GSM8K, lr 3e-6. Training reward rose, but GSM8K dev stayed at 89.4–91.8. Stopped at step 69. Final checks: GSM8K test 90.2 (original checkpoint 90.4); MATH strict +9.6 but MATH acc unchanged. Pod deleted.

## Reproduce

```bash
python src/sweep.py configs/phase1.yaml            # all phase-1 runs (resumes, skips finished)
python src/sweep.py configs/qwen3_17b_baseline.yaml
python src/aggregate.py                            # -> results/dev/summary.csv
```
