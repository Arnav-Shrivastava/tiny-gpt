"""Estimate train/val loss of a checkpoint (same 90/10 split as training).

Example:
    python evaluate.py --eval_iters 50
"""
import argparse

import torch

from model import DEFAULT_CHECKPOINT, DEFAULT_DATA, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tok = load_model(args.checkpoint, args.data, 'cpu')
    with open(args.data, 'r', encoding='utf-8') as f:
        data = torch.tensor(tok.encode(f.read()), dtype=torch.long)
    n = int((1 - args.val_split) * len(data))
    splits = {'train': data[:n], 'val': data[n:]}
    bs = model.config.block_size

    with torch.no_grad():
        for name, d in splits.items():
            losses = []
            for _ in range(args.eval_iters):
                ix = torch.randint(len(d) - bs, (args.batch_size,))
                x = torch.stack([d[i:i + bs] for i in ix])
                y = torch.stack([d[i + 1:i + bs + 1] for i in ix])
                losses.append(model(x, y)[1].item())
            print(f'{name} loss: {sum(losses) / len(losses):.4f} '
                  f'({args.eval_iters} batches of {args.batch_size}x{bs})')


if __name__ == '__main__':
    main()
