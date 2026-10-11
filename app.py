"""Gradio demo for the character-level GPT trained on Bulfinch's Mythology.

Run:  python app.py
"""
import gradio as gr
import torch

from model import load_model

torch.set_num_threads(max(1, torch.get_num_threads()))
model, tok = load_model(device='cpu')  # CPU only: no GPU needed
VOCAB = tok.vocab_size

TITLE = "Tiny GPT: Bulfinch's Mythology"
DESCRIPTION = (
    "Character-level GPT trained from scratch on Bulfinch's Mythology. "
    "**This is a text-completion model, not a chatbot.** It continues the text you give it "
    "in the style of the book, and it does not follow instructions or answer questions reliably."
)
EXAMPLES = [["Zeus"], ["The hero set out"], ["Athena"], ["In the beginning"]]


def complete(prompt, max_new_tokens, temperature, top_k, use_top_k, seed):
    # Characters never seen in training have no token id; drop them.
    ids = tok.encode(prompt or '', skip_unknown=True)
    if not ids:
        ids = tok.encode('\n')
        prompt = ''
    if seed is not None and str(seed) != '':
        torch.manual_seed(int(seed))
    idx = torch.tensor([ids], dtype=torch.long)
    k = int(top_k) if use_top_k else None

    text = prompt
    yield text
    for t in model.generate_stream(idx, int(max_new_tokens), float(temperature), k):
        text += tok.decode(t[0].tolist())
        yield text


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, value="Zeus")
            max_new_tokens = gr.Slider(50, 1000, value=300, step=10, label="Max new tokens (characters)")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            use_top_k = gr.Checkbox(value=True, label="Enable top-k sampling")
            top_k = gr.Slider(1, VOCAB, value=40, step=1, label=f"Top-k (vocab size = {VOCAB})")
            seed = gr.Number(value=None, precision=0, label="Seed (optional, blank = random)")
            run = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Output", lines=22)

    inputs = [prompt, max_new_tokens, temperature, top_k, use_top_k, seed]
    run.click(complete, inputs, output)
    prompt.submit(complete, inputs, output)
    gr.Examples(EXAMPLES, inputs=[prompt])

if __name__ == '__main__':
    demo.queue().launch()
