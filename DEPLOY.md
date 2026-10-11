# Deploying to a Hugging Face Space

I haven't run any of this; run it yourself. Replace `YOUR_USER` and `tinygpt-mythology`.

## 1. Add the Space metadata to README.md

Hugging Face reads configuration from YAML at the very top of `README.md`. Prepend this on the copy that goes to the Space (it also works in the GitHub README, where it renders as a small table):

```yaml
---
title: Tiny GPT Mythology
emoji: 📜
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---
```

## 2. Log in and create the Space

```bash
pip install -U huggingface_hub
huggingface-cli login            # paste a token with write access (newer versions: `hf auth login`)
huggingface-cli repo create tinygpt-mythology --type space --space_sdk gradio
```

## 3a. Recommended: upload files directly (no git history problems)

The 52 MB checkpoint is stored as a normal blob in this repo's git history. Hugging Face requires files over 10 MB to go through Git LFS, so pushing this history as-is will be rejected. Uploading avoids that, and large files are handled for you:

```bash
huggingface-cli upload YOUR_USER/tinygpt-mythology . . --repo-type space \
  --include "app.py" "model.py" "requirements.txt" "README.md" \
            "data/input.txt" "checkpoints/mythology_model.pt"
```

(On newer versions the command is `hf upload ...`.) Only `app.py`, `model.py`, the data, the checkpoint, `requirements.txt` and `README.md` are needed to run the demo.

## 3b. Alternative: git push with LFS (on a fresh clone/branch)

Do this in a separate copy, not in your GitHub repo, so history there stays untouched:

```bash
git clone https://huggingface.co/spaces/YOUR_USER/tinygpt-mythology hf-space
cd hf-space
git lfs install
git lfs track "*.pt"
# copy in: app.py model.py requirements.txt README.md data/ checkpoints/
git add .gitattributes app.py model.py requirements.txt README.md data checkpoints
git commit -m "Add Tiny GPT demo"
git push
```

## 4. After the build

- The Space builds from `requirements.txt` (CPU-only torch via the PyTorch CPU index). The first build takes a few minutes.
- If the build fails on a version pin, loosen `torch`/`gradio` versions in `requirements.txt` to ones your Space's Python supports.
- Put the Space URL in the README's `TODO_HF_SPACE_URL` placeholder.
