from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent

# Load the exact vocabulary used during training.
with open(ROOT / 'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Import the existing model definition safely without re-running training.
spec = spec_from_file_location('tinygpt_dev_v2', ROOT / 'tinygpt-dev-v2.py')
module = module_from_spec(spec)
spec.loader.exec_module(module)
GPTLanguageModel = module.GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load(ROOT / 'mythology_model.pt', map_location=device))
model.eval()


def generate_answer(user_input: str, max_new_tokens: int = 200) -> str:
    prompt = (
        "Question: Who is Jupiter? Answer: Jupiter is the king of the gods. \n"
        "Question: What is Olympus? Answer: Olympus is the mountain home of the deities. \n"
        f"Question: {user_input} \nAnswer:"
    )
    encoded = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(encoded, max_new_tokens=max_new_tokens)

    decoded = decode(generated[0].tolist())
    answer_part = decoded.split('Answer:', 1)[1] if 'Answer:' in decoded else decoded
    return answer_part.strip()


st.set_page_config(page_title="Bulfinch's Mythology Q&A", page_icon="📜")
st.title("Bulfinch's Mythology Q&A")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input("Ask a question about mythology...")
if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.spinner('Thinking...'):
        response = generate_answer(prompt)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
    with st.chat_message('assistant'):
        st.markdown(response)
