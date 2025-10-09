import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import GoRankRNN
from utils import (
    GoSequenceDataset, collate_batch, read_train_split,
)

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    seed_everything(args.seed)

    train_samples, val_samples = read_train_split(args.train_dir)
    assert len(train_samples) > 0, "No training samples parsed. Check train_dir or parser."

    # Fit normalization on train set
    concat = np.concatenate([s['seq'] for s in train_samples], axis=0)
    mean = concat.mean(axis=0, keepdims=True)
    std = concat.std(axis=0, keepdims=True)

    ds_tr = GoSequenceDataset(train_samples, mean=mean, std=std, max_len=args.max_len)
    ds_va = GoSequenceDataset(val_samples, mean=mean, std=std, max_len=args.max_len)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = GoRankRNN(input_dim=ds_tr.samples[0]['seq'].shape[1], hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers, num_classes=9, dropout=args.dropout).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, lengths, yb in dl_tr:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(ds_tr)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, lengths, yb in dl_va:
                xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
                logits = model(xb, lengths)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / max(1, total)
        scheduler.step(acc)

        print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.4f} val_acc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'state_dict': model.state_dict(),
                'mean': ds_tr.mean,
                'std': ds_tr.std,
            }, os.path.join(args.out_dir, 'best_rnn_model.pt'))
            print(f"  > Saved new best with val_acc={best_acc*100:.2f}%")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str, default='./data_set/train_set')
    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    train(args)