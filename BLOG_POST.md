---
title: "MetaMirage: the best LLMs are the worst at knowing when they're wrong"
subtitle: "A paired-task benchmark finds accuracy and metacognition correlate at r = −0.84 across 7 frontier models. Anthropic's own Opus lost 32 TDR points between versions."
date: 2026-04-15
author: Dayeon Kang
tags: [LLM, metacognition, benchmark, AGI, RLHF, Kaggle]
---

# The Best LLMs Are the Worst at Knowing When They're Wrong

**TL;DR:** I built a 50-task metacognition benchmark called MetaMirage. Across 7 frontier LLMs, clean-task accuracy and trap-detection rate correlate at **r = −0.84** (p = 0.018, bootstrap CI [−0.99, −0.58], permutation p = 0.023). The two most accurate models — claude-opus-4-5 at 100% clean accuracy and gpt-4o at 98% — rank **last** on metacognitive monitoring. Within Anthropic alone, **Opus 4.0 → 4.5 lost 32 TDR points** with no accuracy change. The dissociation isn't architectural. It was actively trained in.

---

## What is MetaMirage?

Every LLM benchmark today asks: *did the model get the right answer?* None of them seriously ask: *did the model know when it was about to be wrong?*

That distinction matters. A deployed AGI that confidently answers a misleading question does more harm than one that flags uncertainty. If we optimize for capability without optimizing for monitoring, we're building confident hallucinators and calling it progress.

MetaMirage is built on one idea: **paired tasks**. Each trap is a *clean* variant (genuinely answerable) and a *mirage* variant that looks superficially identical but has a hidden flaw — a false premise, an unanswerable setup, a confidence-lowering context the model has to *notice*. The correct behavior on a mirage is to **name the flaw before answering**, not to answer confidently in spite of it.

Fifty tasks, five families, three scoring modes, seven frontier models. [The repo is here.](https://github.com/dayeon603-pixel/MetaMirage)

## The finding

On global TDR vs. clean accuracy:

| Model | MI | TDR | Clean Acc |
|---|---|---|---|
| claude-haiku-4-5 | **0.615** | 0.756 | 0.963 |
| gpt-4o-mini | 0.574 | 0.845 | 0.759 |
| llama-3-70b | 0.538 | 0.829 | 0.648 |
| claude-sonnet-4-5 | 0.520 | 0.665 | 0.926 |
| gemini-1.5-pro | 0.508 | 0.772 | 0.778 |
| claude-opus-4-5 | **0.409** | 0.555 | **1.000** |
| gpt-4o | 0.407 | 0.626 | 0.982 |

The Metacognitive Index tops out on **claude-haiku-4-5** — the smallest and most recent model in the pool — and bottoms out on claude-opus-4-5 and gpt-4o, the two most accurate models. That's the headline. That's the inversion.

Four independent statistical tests confirm the sign: Student's t p = 0.018, Fisher CI excludes zero, bootstrap CI [−0.99, −0.58] (tighter than Fisher), permutation p = 0.023. All three sign-flip families (`forced_abstention`, `expertise_trap`, `confidence_inversion` — each independently testing a different metacognitive mode) survive leave-one-out stability at min|r| ≥ 0.74.

## The more interesting finding

When I re-ran the benchmark on older Anthropic model versions — which are accessible via explicit version IDs — something striking showed up:

| Lineage | TDR trajectory |
|---|---|
| Opus | 4.0 (0.87) → 4.1 (0.89) → **4.5 (0.55)** |
| Sonnet | 4.0 (0.83) → **4.5 (0.66)** |

**Claude Opus 4.5 lost 32 TDR points relative to Opus 4.0, at constant (actually higher) clean accuracy.** Sonnet lost 16. This is within-vendor, accuracy-controlled. Something changed in the training between Opus 4.1 (August 2025) and Opus 4.5. Whatever that change was, it suppressed metacognitive monitoring.

Claude-haiku-4-5 retained it. The capability/monitoring trade-off isn't a law of scaling. It's a training decision. That's the actionable part.

## Why it probably happens

My best guess: RLHF confidence pressure. Annotators generally prefer decisive answers over hedged ones ([Perez et al. 2022](https://arxiv.org/abs/2212.09251) on sycophancy; [Casper et al. 2023](https://arxiv.org/abs/2307.15217) on reward-model misalignment). That preference, filtered through preference-model training, becomes a signal that penalizes calibrated abstention — the exact behavior MetaMirage rewards.

The mechanistic prediction is falsifiable: if future RLHF datasets incorporate proper-scoring-rule rewards, the sign-flip should weaken generationally. Opus 5.0 will either confirm or refute this.

## What's next

This is a Kaggle submission for the Google DeepMind × Kaggle *Measuring Progress Toward AGI* challenge, Metacognition track. The full 1500-word writeup is on Kaggle as of April 16; the repo is [here](https://github.com/dayeon603-pixel/MetaMirage).

Priorities for v4: more models (contingent on API budget), human baselines, cross-vendor judging, multi-modal extensions. The methodology is public. If you want to throw an LLM at it, the evaluator is one command.

## A nerdy aside on statistics

The point estimate r = −0.84 has a Fisher CI [−0.98, −0.24], which is wide. Honest reading: the true correlation is somewhere from moderate-to-near-perfect negative. But across 36 weight choices for the TDR subscale formula, 94% give r < 0 and 81% clear p < 0.05. Across 10 000 bootstrap resamples, the CI is [−0.99, −0.58] — tighter than Fisher. The sign survives every robustness check I could run.

It's the first time I've seen r cross zero under *any* permutation of the rubric weights — and it doesn't. Not once in ten thousand label shuffles.

---

*This post accompanies the Kaggle submission. Comments, replications, extensions welcome — [GitHub issues](https://github.com/dayeon603-pixel/MetaMirage/issues).*
