# MetaMirage — Kaggle Submission Checklist

**Deadline:** April 17, 2026
**Repo (final state):** https://github.com/dayeon603-pixel/MetaMirage

Print this. Tick boxes as you go. Submit by the morning of April 16 latest, never the evening of April 17.

---

## Pre-submission (do BEFORE opening Kaggle)

- [ ] **Pull latest from GitHub** so your local repo is in sync:
  ```
  cd ~/MetaMirage && git pull origin main
  ```
- [ ] **Read `KAGGLE_WRITEUP.md` end-to-end** one more time. Confirm word count is ≤ 1500:
  ```
  wc -w KAGGLE_WRITEUP.md
  ```
- [ ] **Re-execute the notebook locally** to confirm no errors:
  ```
  python3 -m nbconvert --to notebook --execute kaggle_submission.ipynb --output kaggle_submission.ipynb
  ```
- [ ] **Verify cover_image.png renders** correctly (open in Preview).
- [ ] **Verify all numbers in writeup match `v3_analysis.json`**:
  ```
  python3 -c "import json; a=json.load(open('v3_analysis.json')); print('global r =', a['global_correlation']['tdr_vs_accuracy']['r']); print('haiku MI =', a['profiles']['claude-haiku-4-5-20251001']['metacognitive_index'])"
  ```
  Expect: global r = -0.841, haiku MI = 0.6149

---

## Step 1 — Create the Kaggle Benchmark (private)

The Benchmark is the *project link* requirement. It must be created on Kaggle.com, not just in this repo.

- [ ] Go to https://www.kaggle.com/benchmarks
- [ ] Click "Create Benchmark"
- [ ] **Title:** MetaMirage — Mirage-Pair Metacognition Benchmark
- [ ] **Description:** Paste the first paragraph from `KAGGLE_WRITEUP.md`
- [ ] **Track:** Metacognition
- [ ] **Privacy:** Private (will auto-publish after April 17 deadline)
- [ ] **Upload tasks** — there are two paths:
  - **Path A (preferred):** Run `kaggle_task.py` against the Kaggle SDK to upload via API. Requires `pip install kaggle-benchmarks`.
  - **Path B (fallback):** Manually upload `v3_tasks_50.json` if the Kaggle UI accepts JSON.
- [ ] **Confirm 50 tasks loaded** in the Kaggle Benchmark dashboard.
- [ ] **Note the Benchmark URL** — you'll paste it into the Writeup.

---

## Step 2 — Public Notebook

- [ ] Go to https://www.kaggle.com/code → "+ New Notebook"
- [ ] Click "File" → "Import Notebook" → upload `kaggle_submission.ipynb`
- [ ] **CRITICAL: Run all cells in Kaggle's environment** (Run → Run all). Verify every cell completes without error. If anything fails:
  - Most likely cause: a file path. The notebook expects `v3_analysis.json` and `v3_tasks_50.json` in the working dir. Upload them as Notebook Inputs.
  - If `import anthropic` fails: the notebook should NOT call the API in Kaggle (eval was already run). All API-calling cells are commented out / read from cached JSON. If they aren't, comment them out before saving.
- [ ] **Save the notebook** as a Public Kaggle Notebook.
- [ ] **Note the Notebook URL.**

---

## Step 3 — Create the Writeup

- [ ] Go to https://www.kaggle.com/competitions/measuring-progress-toward-agi-cognitive-abilities → "Submit Writeup"
- [ ] **Title:** MetaMirage: The Sign-Flip Between Capability and Metacognition
- [ ] **Subtitle:** Paste from `KAGGLE_WRITEUP.md` line 3
- [ ] **Track:** Metacognition (must select to submit)
- [ ] **Body:** Paste full content of `KAGGLE_WRITEUP.md`. Verify rendering (tables, code blocks, italic).
- [ ] **Cover image:** Upload `cover_image.png`
- [ ] **Attachments → Add a link:**
  - [ ] Link to your private Kaggle Benchmark (from Step 1)
  - [ ] Link to your Public Notebook (from Step 2)
  - [ ] Optional: link to GitHub repo
- [ ] **Word count check:** Kaggle may show this. Confirm ≤ 1500.

---

## Step 4 — Final review before hitting Submit

- [ ] Subtitle clearly mentions r = −0.84 + Opus regression finding
- [ ] All 8 mandatory sections present (Team, Problem Statement, Task & Benchmark, Dataset, Technical, Results, Affiliations, References)
- [ ] Cover image attached
- [ ] Benchmark link attached and resolves to YOUR benchmark
- [ ] Notebook link attached and resolves to YOUR notebook (and runs)
- [ ] Track = Metacognition selected
- [ ] No personal info in the writeup beyond your name (you're an independent submission)

---

## Step 5 — Submit

- [ ] Hit Submit
- [ ] Verify submission appears in your Kaggle profile under Competitions → Submissions
- [ ] Take a screenshot of the confirmation

---

## Post-submission housekeeping

- [ ] **Rotate API keys** (they were leaked in chat earlier this session):
  - Anthropic: https://console.anthropic.com/settings/keys → revoke the one used + create a new one
  - OpenAI: https://platform.openai.com/api-keys → revoke + create new
- [ ] Delete `~/.metamirage_keys` (contains the leaked keys):
  ```
  rm ~/.metamirage_keys
  ```
- [ ] (Optional) Note the `data/*_records.json` files committed to the public GitHub repo — they contain raw model responses, no API keys. Safe but may want to add to `.gitignore` post-deadline.

---

## If something breaks during submission

| Problem | Fix |
|---|---|
| Notebook cell errors in Kaggle | Comment out the offending cell, re-save, re-run |
| Word count exceeds 1500 | The writeup currently has 42-word headroom; cut from Limitations |
| Cover image rejected (size) | Resize to 1200×630 PNG |
| Benchmark upload fails | Use Path B (manual JSON upload) instead of SDK |
| Kaggle site is slow on deadline | Submit at least 12 hours before the cutoff |

---

## What grading will look at (per official rubric)

| Criterion | Weight | Your defense |
|---|---|---|
| Dataset quality & task construction | 50% | Authored from scratch, novelty audit (max overlap 0.28), item discrimination (10 high-spread items), gold answers human-verified, 3-mode rubric, methodological self-correction documented |
| Novelty / insights / discriminatory power | 30% | Sign-flip (4-test confirmed), within-Anthropic generational regression (Opus 4.5 lost 32 TDR pts), haiku counterexample, MI spread 0.208, mechanistic RLHF hypothesis |
| Writeup quality | 20% | All 8 mandatory sections, pre-registered hypothesis framing, mechanistic theory with citations, honest limitations |

You're done. Good luck.
