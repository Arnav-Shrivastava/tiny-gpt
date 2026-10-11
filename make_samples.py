"""Generate a samples markdown file from a checkpoint (real, unedited outputs).

Example:
    python make_samples.py --checkpoint checkpoints/mythology_model_v2.pt --out assets/samples.md
"""
import argparse

import torch

from model import DEFAULT_CHECKPOINT, load_model

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
p.add_argument('--out', default='assets/samples.md')
args = p.parse_args()

model, tok = load_model(args.checkpoint)
configs = [('Zeus', 0.5, 40, 1), ('The hero set out', 0.8, 40, 2),
           ('Athena', 1.0, None, 3), ('In the beginning', 1.2, None, 4)]
FENCE = '`' * 3
out = ['# Sample outputs\n',
       f'Raw outputs from `{args.checkpoint.replace(chr(92), "/")}`, 300 new characters each, unedited. '
       'Reproduce with `python generate.py --checkpoint <ckpt> --prompt P --max_new_tokens 300 '
       '--temperature T [--top_k K] --seed S`.\n']
for prompt, temp, k, seed in configs:
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)])
    text = prompt + ''.join(tok.decode(x[0].tolist()) for x in model.generate_stream(idx, 300, temp, k))
    out.append(f'## Prompt: "{prompt}" (temperature={temp}, top_k={k or "off"}, seed={seed})\n\n'
               f'{FENCE}text\n{text}\n{FENCE}\n')
with open(args.out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
