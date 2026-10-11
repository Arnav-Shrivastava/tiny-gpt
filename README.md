# Tiny GPT: a character-level GPT trained on Bulfinch's Mythology

A ~10.8M-parameter decoder-only Transformer, written in PyTorch and trained from scratch on Bulfinch's *Mythology*. [Live demo](TODO_HF_SPACE_URL)

<!-- TODO: add a demo screenshot at assets/demo.png, then uncomment -->
<!-- ![Demo screenshot](assets/demo.png) -->

## What it is (and isn't)

- **Is:** a base language model. Given a prompt, it predicts the next character, over and over, so it continues the text in the style of the book.
- **Is not:** a chatbot or an assistant. It has no instruction tuning, doesn't follow instructions, and doesn't answer questions reliably. Output is book-flavoured, locally plausible text with invented words and facts.

## Model

| Setting | Value |
|---|---|
| Type | Decoder-only Transformer, pre-LayerNorm, ReLU MLP |
| Layers | 6 |
| Attention heads | 6 (head size 64) |
| Embedding size (`n_embd`) | 384 |
| Context length (`block_size`) | 256 characters |
| Vocabulary | 90 characters (character-level, no BPE) |
| Dropout | 0.2 |
| Parameters | 10,808,154 (10.81M) |
| Checkpoint | `state_dict`, 52.75 MB (includes causal-mask buffers) |

Training setup (from `experiments/02_gpt_v2.py`, defaults of `train.py`): AdamW, learning rate 3e-4, batch size 64, 5000 iterations, seed 1337.

## Dataset

Bulfinch's *Mythology* (public domain), in `data/input.txt`: 839,083 characters, 90 unique. Split 90/10 by position: 755,174 characters train, 83,909 validation.

## Results

Loss measured on the shipped checkpoint with `python evaluate.py --eval_iters 40 --batch_size 32` (seed 1337, random 256-character windows):

| Split | Cross-entropy loss |
|---|---|
| Train | 0.7035 |
| Val | 2.2389 |

The large gap shows **heavy overfitting**: the model has memorised much of the small training text. No training-time loss log was saved for this checkpoint, so there is no loss curve yet. Re-run training (see below) and then `python plot_loss.py` to produce `assets/loss_curve.png`. Note the evaluation above assumes the checkpoint was trained on the same 90/10 split as the dev script.

## Sample outputs

Unedited, 300 new characters each (full set and exact settings in [assets/samples.md](assets/samples.md)). Excerpt, prompt "Zeus", temperature 0.5, top-k 40:

```text
Zeus was done in the death of the cave, and the other animals who
consumed the lamentations of the earth. He was called the
Golden Fleece, which proceeded to the place of Apollo and Medea
Charlemagnelia, and descended to be considered as a rock to the
contest. A story of Perseus and Psyche in the air we
```

Prompt "The hero set out", temperature 0.8, top-k 40:

```text
The hero set out upon his body to the woods to return
to Achilles' field, and the field entered by the aid of the most
celebrated subjects from the combatants of his mistress. He had
a tedion of the ships of Troy the Belvedere Achilles with the Hellenic
to Ajax, and at the Eurystheus suggesting the Furies.
```

Prompt "In the beginning", temperature 1.2, no top-k (note the degeneration at high temperature):

```text
In the beginning verse,
    Raising all joy Egypt at the hunted of the ancient beast."

THE AND ALATON

That was a cost widen, descended from Ceyx, covered by Philoctetes,
```

## Quickstart

```bash
pip install -r requirements.txt

# Generate text from the included checkpoint
python generate.py --prompt "Zeus" --max_new_tokens 300 --temperature 0.8 --top_k 40

# Run the Gradio demo (CPU is fine)
python app.py

# Retrain (writes a new checkpoint + CSV loss log; refuses to overwrite without --force)
python train.py --out checkpoints/run2.pt
pip install -r requirements-dev.txt
python plot_loss.py --log checkpoints/run2_loss.csv --out assets/loss_curve.png

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/run2.pt
```

Training the default config on CPU is slow (hours); a GPU is recommended.

## Project structure

```
app.py                 Gradio demo (streaming, temperature, top-k, seed)
model.py               GPT model, config, character tokenizer, checkpoint loader
train.py               Training script (CLI args, CSV loss log)
generate.py            CLI text generation
evaluate.py            Train/val loss of a checkpoint
plot_loss.py           Plot loss curve from a train.py log
make_samples.py        Regenerates assets/samples.md
data/input.txt         Training corpus
checkpoints/           Trained weights
notebooks/             Original development notebook
experiments/           01_bigram.py, 02_gpt_v2.py (the original step-by-step scripts)
assets/                Sample outputs (and loss curve once generated)
DEPLOY.md              Hugging Face Spaces deployment steps
```

## What I learned / limitations

- **Character-level tokenization** keeps the vocabulary tiny but makes sequences long and forces the model to learn spelling; it invents plausible non-words ("tedion", "motionals").
- **Tiny data** (about 0.84M characters) is far too little for 10.8M parameters; the train/val gap above is the result.
- **Overfitting:** validation loss is much higher than train loss; more data, more dropout, or early stopping would help.
- **No instruction following:** it only continues text. Prompts like questions get book-style continuations, not answers.
- Coherence is local (a sentence or two) and long-range facts are unreliable.

## Future work

- BPE tokenizer instead of characters.
- Ablations on context length and number of layers.
- Save per-run loss logs and use early stopping / checkpoint selection on validation loss.
- Fine-tune on a small instruction set.

## Credits

Built following Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and his [nanoGPT](https://github.com/karpathy/nanoGPT).
