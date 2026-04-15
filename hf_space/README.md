---
title: MetaMirage
emoji: 🪞
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# MetaMirage — Interactive Demo

Paired-task metacognition benchmark. [Main repo](https://github.com/dayeon603-pixel/MetaMirage).

## Deploy steps

1. Create a new Space at https://huggingface.co/new-space — SDK = Gradio, hardware = CPU.
2. Clone your Space repo locally:
   ```
   git clone https://huggingface.co/spaces/YOUR_USERNAME/MetaMirage
   ```
3. Copy these files in:
   ```
   cp app.py requirements.txt README.md v3_tasks_50.json eval_system_prompt.txt YOUR_SPACE_DIR/
   ```
4. Push:
   ```
   cd YOUR_SPACE_DIR && git add -A && git commit -m "init" && git push
   ```
5. Wait ~2 min for the Space to build. It'll be live at
   `https://huggingface.co/spaces/YOUR_USERNAME/MetaMirage`.

Paste that URL into the Kaggle writeup "Attachments → Add link" section.

## What it does

Visitor pastes an Anthropic API key, picks a Claude model, and runs 5-20
tasks from the MetaMirage benchmark. Results are shown against the published
leaderboard so visitors can see where their tested model falls.

No API keys are stored server-side.
