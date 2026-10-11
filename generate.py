"""Generate text from a trained checkpoint.

Example:
    python generate.py --prompt "Zeus" --max_new_tokens 300 --temperature 0.8 --top_k 40
"""
import argparse
import sys

import torch

from model import DEFAULT_CHECKPOINT, DEFAULT_DATA, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--data', default=DEFAULT_DATA, help='corpus used to rebuild the character vocabulary')
    p.add_argument('--prompt', default='\n', help='text to continue (default: a newline)')
    p.add_argument('--max_new_tokens', type=int, default=300)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0, help='0 disables top-k filtering')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok = load_model(args.checkpoint, args.data, 'cpu')
    ids = tok.encode(args.prompt, skip_unknown=True)
    if not ids:
        sys.exit('Prompt contains no characters from the training vocabulary.')
    idx = torch.tensor([ids], dtype=torch.long)

    print(args.prompt, end='', flush=True)
    for tokn in model.generate_stream(idx, args.max_new_tokens, args.temperature, args.top_k or None):
        print(tok.decode(tokn[0].tolist()), end='', flush=True)
    print()


if __name__ == '__main__':
    main()
