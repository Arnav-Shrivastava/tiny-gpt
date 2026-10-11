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
    if spec is None or spec.loader is None:
        raise ImportError('Could not load tinygpt-dev-v2.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    m = module.GPTLanguageModel().to(device)
    state = torch.load(os.path.join(ROOT, 'mythology_model.pt'), map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m


model = load_model()

FEW_SHOT = (
    "Question: Who is Jupiter? \nAnswer: Jupiter is the king of the gods. \n\n"
    "Question: What is Olympus? \nAnswer: Olympus is the mountain home of the deities. \n\n"
    "Question: Who is Apollo? \nAnswer: Apollo is the god of the sun, music, and poetry. \n\n"
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
if 'pending' not in st.session_state:
    st.session_state.pending = None

n_params = sum(p.numel() for p in model.parameters())

with st.sidebar:
    st.header("About this Project")
    st.markdown("🛠️ **Built from scratch by Arnav Shrivastava**")
    st.markdown(
        "- Custom **PyTorch Transformer** (GPT-style, decoder-only)\n"
        "- **Character-level** next-token prediction\n"
        f"- **~{n_params / 1e6:.1f}M parameters**, vocab size {vocab_size}\n"
        "- Trained from scratch on *Bulfinch's Mythology*\n"
        "- Q&A format achieved via hidden few-shot prompting"
    )
    st.caption("Base model, not instruction-tuned: answers are in the book's style and may be imprecise.")
    st.divider()
    st.subheader("Try it out")
    for q in ["Who is Zeus?", "What is Olympus?", "Who was Hercules?"]:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_msg = st.chat_input("Ask a question about mythology...")
if st.session_state.pending:
    user_msg = st.session_state.pending
    st.session_state.pending = None

if user_msg:
    st.session_state.messages.append({'role': 'user', 'content': user_msg})
    with st.chat_message('user'):
        st.markdown(user_msg)

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_answer(user_msg)
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
