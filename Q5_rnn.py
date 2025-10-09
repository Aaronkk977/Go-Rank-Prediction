"""
This script MUST generate submission.csv when executed, per the assignment spec.
Usage:
python Q5_rnn.py --test_dir ./test --weights ./outputs/best_rnn_model.pt --out submission_rnn.csv


If you want to re-train from scratch inside Q5.py, add a flag like --train_first (NOT default).
"""
import os
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import GoRankRNN
from utils import (
GoSequenceDataset, collate_batch, read_test_files, ID_TO_RANK
)


def load_model(weights_path: str, input_dim: int = 31, hidden_dim: int = 128,
               num_layers: int = 2, dropout: float = 0.2, device='cpu'):
    # 允許非 weights-only 的反序列化（信任來源時可用）
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    model = GoRankRNN(input_dim=input_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, num_classes=9, dropout=dropout)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    mean = ckpt['mean']
    std  = ckpt['std']
    return model, mean, std


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model, mean, std = load_model(args.weights, input_dim=args.input_dim, hidden_dim=args.hidden_dim,
    num_layers=args.num_layers, dropout=args.dropout, device=device)


    test_items = read_test_files(args.test_dir) # list[(file_id, seq[T,F])]


    # Build dataset and loader (labels are dummy -1)
    samples = [{'seq': seq, 'y': -1} for _, seq in test_items]
    ds_te = GoSequenceDataset(samples, mean=mean, std=std, max_len=args.max_len)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)


    preds = []
    file_ids = [fid for fid, _ in test_items]


    with torch.no_grad():
        offset = 0
        for xb, lengths, _ in dl_te:
            xb, lengths = xb.to(device), lengths.to(device)
            logits = model(xb, lengths)
            pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred_ids)
            offset += xb.size(0)


    # Write submission
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for fid, pid in zip(file_ids, preds):
            rank_num = int(pid) + 1
            writer.writerow([int(fid), rank_num])  
    print(f"Saved submission to {out_path} with {len(preds)} rows.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', type=str, default='./data_set/test_set')
    ap.add_argument('--weights', type=str, default='./outputs/best_rnn_model.pt')
    ap.add_argument('--out', type=str, default='submission_rnn.csv')


    # Model hyper-params should match your trained checkpoint
    ap.add_argument('--input_dim', type=int, default=31)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)


    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--cpu', action='store_true')


    args = ap.parse_args()
    predict(args)