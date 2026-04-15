# Tweet Thread — MetaMirage launch

Post this if you have an X/Twitter account. One-shot thread, 8 tweets. Copy each
block below (each starts with "🧵 i/N"), post sequentially.

---

**1/8**
🧵 1/8

I built a metacognition benchmark for LLMs. The finding surprised me.

Across 7 frontier models, clean-task accuracy and trap-detection rate correlate at **r = −0.84** (p = 0.018, bootstrap CI [−0.99, −0.58]).

The most accurate LLMs are the worst at knowing when they're wrong.

---

**2/8**

Design: 50 paired tasks. Each trap has a *clean* variant (answerable) and a *mirage* variant that looks identical but has a hidden flaw — false premise, unanswerable setup, confidence-lowering context the model has to notice.

Correct behavior on a mirage: name the flaw before answering.

---

**3/8**

Leaderboard (Metacognitive Index):

1. claude-haiku-4-5  — MI 0.615 (acc 96%)
2. gpt-4o-mini       — MI 0.574
3. llama-3-70b       — MI 0.538
4. claude-sonnet-4-5 — MI 0.520
5. gemini-1.5-pro    — MI 0.508
6. claude-opus-4-5   — MI 0.409 (acc 100%)
7. gpt-4o            — MI 0.407 (acc 98%)

The top-accuracy models rank **last**.

---

**4/8**

Four independent stats tests confirm the sign: Student's t (p=0.018), Fisher CI, bootstrap CI [−0.99, −0.58], permutation p=0.023.

Three of five task families independently show the sign-flip. LOO-stable across all 7 folds.

---

**5/8**

The more interesting finding: within Anthropic, **Opus 4.0 → 4.5 lost 32 TDR points**, with no accuracy change.

Opus 4.0: TDR 0.87, acc 0.98
Opus 4.1: TDR 0.89, acc 0.96
Opus 4.5: **TDR 0.55**, acc 1.00

Sonnet 4.0 → 4.5: TDR 0.83 → 0.66 (-16 pts).

Same lab, accuracy held constant, monitoring collapsed.

---

**6/8**

Claude-haiku-4-5 retained monitoring. So the trade-off isn't architectural — it was trained in.

Best guess for mechanism: RLHF confidence pressure. Annotators prefer decisive answers (Perez 2022, Casper 2023) → RLHF penalizes calibrated abstention → larger, more-RLHF'd models overfit to confidence.

---

**7/8**

Falsifiable prediction: if future RLHF training incorporates proper scoring rules (Bayesian calibration rewards), the sign-flip should weaken in later generations.

Opus 5.0 will confirm or refute this.

---

**8/8**

This is a submission for @GoogleDeepMind × @Kaggle *Measuring Progress Toward AGI* (Metacognition track). Full writeup + code:

- GitHub: github.com/dayeon603-pixel/MetaMirage
- Demo: [HF SPACE URL ONCE DEPLOYED]
- Blog post: [GITHUB PAGES URL ONCE PUBLISHED]

Submitted by a high schooler. Replications welcome. /end
