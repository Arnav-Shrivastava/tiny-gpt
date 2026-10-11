"""Train the character-level GPT.

Defaults reproduce the hyperparameters of experiments/02_gpt_v2.py.

Example:
    python train.py --out checkpoints/run2.pt
    python train.py --max_iters 20 --eval_interval 10 --eval_iters 5 --out tmp/smoke.pt
"""
import argparse
import csv
import os
import time

import torch

from model import DEFAULT_DATA, CharTokenizer, GPTConfig, GPTLanguageModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--out', default='checkpoints/mythology_model.pt', help='checkpoint path (state_dict)')
    p.add_argument('--log', default=None, help='loss log CSV (default: <out>_loss.csv)')
    p.add_argument('--force', action='store_true', help='allow overwriting an existing checkpoint')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--max_iters', type=int, default=5000)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--n_embd', type=int, default=384)
    p.add_argument('--n_head', type=int, default=6)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f'{args.out} already exists. Choose another --out or pass --force.')
    log_path = args.log or os.path.splitext(args.out)[0] + '_loss.csv'
    for path in (args.out, log_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int((1 - args.val_split) * len(data))
    splits = {'train': data[:n], 'val': data[n:]}
    print(f'{len(text):,} chars, vocab {tok.vocab_size}, train {n:,} / val {len(data) - n:,}, device {device}')

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    config = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
                       n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
    model = GPTLanguageModel(config).to(device)
    print(f'{model.num_params() / 1e6:.2f}M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                losses[k] = model(X, Y)[1].item()
            out[split] = losses.mean().item()
        model.train()
        return out

    start = time.time()
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss'])
        for it in range(args.max_iters):
            if it % args.eval_interval == 0 or it == args.max_iters - 1:
                losses = estimate_loss()
                writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}"])
                f.flush()
                print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} "
                      f'({time.time() - start:.0f}s)')
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args.out)
    print(f'Saved checkpoint to {args.out}\nLoss log: {log_path}')


if __name__ == '__main__':
    main()
