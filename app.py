import os
from importlib.util import module_from_spec, spec_from_file_location

import streamlit as st
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Vocabulary (must match training exactly)
# ---------------------------------------------------------------------------
with open(os.path.join(ROOT, 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # The filename contains hyphens, so load it by path. Its training code is
    # guarded by __main__, so importing does not retrain.
    spec = spec_from_file_location('tinygpt_dev_v2', os.path.join(ROOT, 'tinygpt-dev-v2.py'))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    m = module.GPTLanguageModel().to(device)
    state = torch.load(os.path.join(ROOT, 'mythology_model.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m


model = load_model()

FEW_SHOT = (
    "Question: Who is Jupiter? Answer: Jupiter is the king of the gods. \n"
    "Question: What is Olympus? Answer: Olympus is the mountain home of the deities. \n"
)


def generate_answer(user_input: str, max_new_tokens: int = 200) -> str:
    prompt = FEW_SHOT + f"Question: {user_input} \nAnswer:"
    # Drop characters never seen in training to avoid KeyError.
    ids = [stoi[c] for c in prompt if c in stoi]
    context = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_new_tokens)

    # Slice off the prompt by token count: only newly generated tokens remain,
    # so the few-shot context can never leak into the UI.
    answer = decode(out[0].tolist()[len(ids):])

    # The model will keep inventing further Q&A pairs; stop at the next one.
    answer = answer.split("\nQuestion:", 1)[0].split("Question:", 1)[0]
    return answer.strip() or "..."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bulfinch's Mythology Q&A", page_icon="📜")
st.title("Bulfinch's Mythology Q&A")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_msg := st.chat_input("Ask a question about mythology..."):
    st.session_state.messages.append({'role': 'user', 'content': user_msg})
    with st.chat_message('user'):
        st.markdown(user_msg)

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_answer(user_msg)
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
