"""Plot train/val loss from a CSV written by train.py.

Example:
    python plot_loss.py --log checkpoints/mythology_model_v2_loss.csv --out assets/loss_curve.png
"""
import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log', default='checkpoints/mythology_model_v2_loss.csv')
    p.add_argument('--out', default='assets/loss_curve.png')
    args = p.parse_args()

    steps, train, val = [], [], []
    with open(args.log, newline='') as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            train.append(float(row['train_loss']))
            val.append(float(row['val_loss']))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, train, marker='o', label='train')
    plt.plot(steps, val, marker='o', label='val')
    plt.xlabel('step')
    plt.ylabel('cross-entropy loss')
    plt.title('Tiny GPT training loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
