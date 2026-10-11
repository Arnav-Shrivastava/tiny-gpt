"""Generate assets/samples.md from the checkpoint (real, unedited outputs)."""
import torch

from model import load_model

model, tok = load_model()
configs = [('Zeus', 0.5, 40, 1), ('The hero set out', 0.8, 40, 2),
           ('Athena', 1.0, None, 3), ('In the beginning', 1.2, None, 4)]
FENCE = '`' * 3
out = ['# Sample outputs\n',
       'Raw outputs from `checkpoints/mythology_model.pt`, 300 new characters each, unedited. '
       'Reproduce with `python generate.py --prompt P --max_new_tokens 300 --temperature T '
       '[--top_k K] --seed S`.\n']
for prompt, temp, k, seed in configs:
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)])
    text = prompt + ''.join(tok.decode(x[0].tolist()) for x in model.generate_stream(idx, 300, temp, k))
    out.append(f'## Prompt: "{prompt}" (temperature={temp}, top_k={k or "off"}, seed={seed})\n\n'
               f'{FENCE}text\n{text}\n{FENCE}\n')
with open('assets/samples.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
