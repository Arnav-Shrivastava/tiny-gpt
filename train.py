"""Train the character-level GPT.

Defaults reproduce the architecture and hyperparameters of experiments/02_gpt_v2.py,
plus best-checkpoint saving and early stopping on validation loss.

Only the checkpoint with the lowest validation loss seen so far is written to --out;
final-iteration weights are NOT saved unless they are the best.

Example:
    python train.py                       # writes checkpoints/mythology_model_v2.pt
    python train.py --max_iters 20 --eval_interval 10 --eval_iters 3 --out tmp/smoke.pt
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
    p.add_argument('--out', default='checkpoints/mythology_model_v2.pt',
                   help='best checkpoint path (state_dict)')
    p.add_argument('--log', default=None, help='loss log CSV (default: <out>_loss.csv)')
    p.add_argument('--force', action='store_true', help='allow overwriting an existing checkpoint')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--max_iters', type=int, default=5000)
    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--patience', type=int, default=6,
                   help='stop after this many evals without val improvement (0 disables)')
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
    if device == 'cuda':
        print(f'Device: cuda ({torch.cuda.get_device_name(0)})')
    else:
        print('Device: cpu (CUDA not available)')

    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int((1 - args.val_split) * len(data))
    splits = {'train': data[:n], 'val': data[n:]}
    print(f'{len(text):,} chars, vocab {tok.vocab_size}, train {n:,} / val {len(data) - n:,}')

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

    best_val, best_step, bad_evals = float('inf'), -1, 0
    start = time.time()
    stopped_early = False
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'best'])
        for it in range(args.max_iters):
            if it % args.eval_interval == 0 or it == args.max_iters - 1:
                losses = estimate_loss()
                improved = losses['val'] < best_val
                if improved:
                    best_val, best_step, bad_evals = losses['val'], it, 0
                    torch.save(model.state_dict(), args.out)
                else:
                    bad_evals += 1
                writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", int(improved)])
                f.flush()
                print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                      f"{' *saved*' if improved else ''} ({time.time() - start:.0f}s)")
                if args.patience and bad_evals >= args.patience:
                    print(f'Early stopping: no val improvement for {bad_evals} evals.')
                    stopped_early = True
                    break
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    print(f'Best val loss {best_val:.4f} at step {best_step}'
          f"{' (stopped early)' if stopped_early else ''}.\n"
          f'Best checkpoint: {args.out}\nLoss log: {log_path}')


if __name__ == '__main__':
    main()
