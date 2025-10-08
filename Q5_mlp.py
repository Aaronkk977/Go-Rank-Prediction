#!/usr/bin/env python3
"""
Q5_mlp.py — Go Rank Prediction with MLP (DeepSets-style)

This script matches the I/O style of the user's RNN version: it ingests per-file
sequences of per-move features with variable length, but uses a permutation-
invariant MLP+pooling architecture (DeepSets) instead of RNN.

Usage examples
--------------
# Train then generate submission.csv to ./submission.csv
python Q5_mlp.py --data_dir /path/to/dataset --train --epochs 12 --out submission.csv

# Only inference (expects a pretrained weight file)
python Q5_mlp.py --data_dir /path/to/dataset --weights weights_mlp.pt --out submission.csv

Directory layout (same as assignment spec)
------------------------------------------
<data_dir>
  train/
    log_1D_policy_train.txt
    ...
    log_9D_policy_train.txt
  test/
    1.txt
    2.txt
    ...

Output format
-------------
CSV with two columns: id,label (label is an integer 1..9 corresponding to 1D..9D)

Implementation notes
--------------------
- Parser follows the assignment description: each move has 31 numeric fields
  in 5 lines after the color+coord line. We compute a set of robust per-move
  features and feed them to a per-move MLP (phi). Then we aggregate over all
  moves in the file with pooling (mean + max + logsumexp) and an attention
  pooling head. The aggregated vector is classified by an MLP (rho).
- Training uses stratified sampling across ranks, label smoothing and Cosine LR.
- Inference runs on CPU by default.

Author: Rubin + ChatGPT (MLP variant)
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Parsing utilities
# ---------------------------
MOVE_RE = re.compile(r"^[BW]\[[A-T][0-9]{1,2}\]")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
PCT_RE = re.compile(r"([-+]?\d*\.?\d+)\s*%")

# As per PDF spec: 9 policy + 9 values + 9 rankouts + 1 strength + 3 kata = 31
RAW_DIM = 31

# We add some light engineered scalars (argmaxes, top-k means, deltas, color sign)
# Final per-move feature dim (F): RAW_DIM + 8 engineered = 39 by default
EXTRA_DIM = 8
FEAT_DIM = RAW_DIM + EXTRA_DIM

RANKS = ["1D","2D","3D","4D","5D","6D","7D","8D","9D"]
RANK_TO_INT = {r:i+1 for i,r in enumerate(RANKS)}  # '1D'->1
INT_TO_RANK = {i+1:r for i,r in enumerate(RANKS)}  # 1->'1D'


def _to_floats_from_line(line: str) -> List[float]:
    # Correctly handles lines with mixed percentages and floats.
    # It works by first substituting all percentages (e.g., "41.2%")
    # with their decimal representation ("0.412") in the string.
    
    def percentage_to_float_string(match):
        # Takes a regex match object, converts the captured number, and returns it as a string.
        return str(float(match.group(1)) / 100.0)

    # Use re.sub with a function to perform the replacement for all occurrences.
    # For a line "41.2% -0.6 17.2", this creates a new string "0.412 -0.6 17.2"
    processed_line = PCT_RE.sub(percentage_to_float_string, line)
    
    # Now, safely find all float numbers in the processed string.
    return [float(x) for x in FLOAT_RE.findall(processed_line)]


def _parse_one_move(block: List[str]) -> Optional[List[float]]:
    """Parse a 6-line move block into a FEAT_DIM vector.
    block[0]: 'B[Q16]' or 'W[Q16]'
    block[1]: 9 policy
    block[2]: 9 value (winrate)
    block[3]: 9 rank model outputs
    block[4]: 1 strength score
    block[5]: 3 kata: winrate(black), lead(black), uncertainty
    For white moves, invert the signs of winrate/lead as per spec.
    """
    if len(block) < 6 or not MOVE_RE.match(block[0].strip()):
        return None

    color = 1.0 if block[0].startswith("B[") else -1.0

    policy = np.asarray(_to_floats_from_line(block[1]), dtype=np.float32)
    values = np.asarray(_to_floats_from_line(block[2]), dtype=np.float32)
    rankouts = np.asarray(_to_floats_from_line(block[3]), dtype=np.float32)

    if policy.size != 9 or values.size != 9 or rankouts.size != 9:
        return None

    strength = float(_to_floats_from_line(block[4])[0]) if _to_floats_from_line(block[4]) else 0.0

    kata = _to_floats_from_line(block[5])
    if len(kata) < 2:
        return None
    winrate_b = float(kata[0])
    lead_b = float(kata[1])
    uncert = float(kata[2]) if len(kata) > 2 else 0.0

    # color correction: convert to current player's perspective
    # Spec notes both are from black's perspective
    winrate = winrate_b if color > 0 else (1.0 - winrate_b)
    lead = lead_b if color > 0 else -lead_b

    # Engineered scalars
    pol_argmax = float(policy.argmax()) / 8.0
    val_argmax = float(values.argmax()) / 8.0
    rank_argmax = float(rankouts.argmax()) / 8.0
    pol_top5 = float(np.sort(policy)[-5:].mean())
    val_top5 = float(np.sort(values)[-5:].mean())
    rank_peak = float(rankouts.max())
    # Consistency between value and rankouts center (5D index=4)
    center_score = float(values[4]) * float(rankouts[4])
    # Simple uncertainty-aware value
    val_unc = float(winrate) * math.exp(-float(uncert))

    feats = np.concatenate([
        policy, values, rankouts,
        np.array([strength, winrate, lead, uncert], dtype=np.float32),
        np.array([pol_argmax, val_argmax, rank_argmax, pol_top5, val_top5,
                  rank_peak, center_score, val_unc], dtype=np.float32)
    ], axis=0)
    assert feats.shape[0] == FEAT_DIM
    return feats.tolist()


def split_games(lines: List[str]) -> List[List[str]]:
    games: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if ln.strip().startswith("Game "):
            if cur:
                games.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        games.append(cur)
    return games

def parse_game_moves(game_lines: List[str]) -> List[List[float]]:
    moves: List[List[float]] = []
    i = 0
    n = len(game_lines)
    while i + 5 < n:
        block = game_lines[i:i+6]
        vec = _parse_one_move(block)
        if vec is not None:
            moves.append(vec)
            i += 6
        else:
            i += 1
    return moves

def parse_file_moves_from_path(txt_path: Path, is_train: bool) -> List[List[List[float]]]:
    lines = Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if is_train:
        games = split_games(lines)
        return [parse_game_moves(g) for g in games]
    else:
        games = split_games(lines)
        if len(games) == 0:
            return [parse_game_moves(lines)]
        else:
            return [parse_game_moves(g) for g in games]

# ---------------------------
# Dataset
# ---------------------------
class GoRankDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.split = split
        self.samples: List[Tuple[str, np.ndarray, Optional[int]]] = []
        data_root = Path(data_dir)
        if split == "train":
            train_dir = data_root / "train_set"
            if not train_dir.exists():
                raise FileNotFoundError(f"Train directory not found: {train_dir}")
            pattern_files = [f for f in os.listdir(train_dir) if f.endswith('.txt') and 'train' in f]
            for fname in sorted(pattern_files):
                m = re.search(r"([1-9])D", fname)
                if not m:
                    continue
                k = int(m.group(1))  # 1..9
                p = train_dir / fname
                games = parse_file_moves_from_path(p, is_train=True)
                # print a little info
                print(f"[INFO] Parsed {len(games)} games from {fname} with rank {k}D")

                gid = 0
                for mvlist in games:
                    if len(mvlist) == 0:
                        continue
                    X = np.asarray(mvlist, dtype=np.float32)
                    y = k
                    gid += 1
                    fid = f"{p.name}#G{gid}"
                    self.samples.append((fid, X, y))
            if len(self.samples) == 0:
                present = os.listdir(train_dir)
                raise ValueError(
                    "No training samples parsed. Ensure files contain 'train' in name and ranks like '1D'..'9D'."
                    f"Train dir: {train_dir} Present files: {present}"
                )
        else:
            test_dir = data_root / "test_set"
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            for fname in sorted(os.listdir(test_dir)):
                if not fname.endswith('.txt'):
                    continue
                p = test_dir / fname
                games = parse_file_moves_from_path(p, is_train=False)
                mv = [m for g in games for m in g]
                X = np.asarray(mv, dtype=np.float32)
                self.samples.append((fname, X, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fid, X, y = self.samples[idx]
        return fid, torch.from_numpy(X), (0 if y is None else y)


def pad_collate(batch):
    # batch: list of (fid, [T,F], y)
    fids, seqs, ys = zip(*batch)
    # Ensure each sequence has at least length 1 (all-zero placeholder if empty)
    fixed = []
    for s in seqs:
        if s.ndim == 2 and s.shape[0] == 0:
            fixed.append(torch.zeros(1, FEAT_DIM, dtype=torch.float32))
        else:
            fixed.append(s)
    seqs = fixed

    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    maxT = max([s.shape[0] for s in seqs])
    Fdim = seqs[0].shape[1] if maxT > 0 else FEAT_DIM
    xt = torch.zeros(len(seqs), maxT, Fdim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s.shape[0] > 0:
            xt[i, :s.shape[0], :s.shape[1]] = s
    y = torch.tensor(ys, dtype=torch.long)
    return fids, xt, lengths, y


# ---------------------------
# Model: DeepSets-style MLP + pooling
# ---------------------------
class MoveEncoder(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hid=128, out=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, out), nn.ReLU(), nn.Dropout(p)
        )
    def forward(self, x):  # x: [B,T,F]
        B,T,Fd = x.shape
        x = self.net(x)
        return x  # [B,T,out]

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, h, mask):
        # h: [B,T,d], mask: [B,T] (True for valid)
        scores = self.w(h).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(~mask, -1e9)
        a = F.softmax(scores, dim=1).unsqueeze(-1)  # [B,T,1]
        return (h * a).sum(dim=1)  # [B,d]

class GoRankMLP(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, d=128, p=0.1, num_classes=9):
        super().__init__()
        self.enc = MoveEncoder(in_dim, hid=d, out=d, p=p)
        self.attn = AttnPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d*4),
            nn.Linear(d*4, d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d, num_classes)
        )
    def forward(self, x, lengths):
        # x: [B,T,F]
        B,T,Fd = x.shape
        mask = torch.arange(T, device=x.device)[None,:] < lengths[:,None]
        h = self.enc(x)
        # pooling: mean / max / logsumexp + attention
        mask_f = mask.float().unsqueeze(-1)
        h_masked = h * mask_f
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (h_masked.sum(dim=1) / denom)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        lse_pool = torch.logsumexp(h.masked_fill(~mask.unsqueeze(-1), -1e9), dim=1)
        attn_pool = self.attn(h, mask)
        agg = torch.cat([mean_pool, max_pool, lse_pool, attn_pool], dim=1)
        logits = self.head(agg)
        return logits


# ---------------------------
# Training / Evaluation
# ---------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        # targets: [B] in 1..9
        targets = targets - 1
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        onehot = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
        soft = (1 - self.eps) * onehot + self.eps / n
        loss = -(soft * logp).sum(dim=1).mean()
        return loss


def train_one_epoch(model, loader, opt, sched, device):
    model.train()
    crit = LabelSmoothingCE(0.05)
    total, correct, loss_sum = 0, 0, 0.0
    for _, x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1) + 1
            total += y.numel()
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.numel()
    return loss_sum/total, correct/total


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for _, x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            pred = logits.argmax(dim=1) + 1
            total += y.numel()
            correct += (pred == y).sum().item()
    return correct/total if total>0 else 0.0


def run_train(data_dir: str, weights_out: str, batch_size=8, epochs=12, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = GoRankDataset(data_dir, split="train")
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Parsed 0 training samples from {data_dir}/train. Please check filenames and content.")
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr = idx[: max(1, int(0.8*n))]
    va = idx[max(1, int(0.8*n)):]

    def subset(ds, ids):
        class _Sub(Dataset):
            def __init__(self, base, ids):
                self.base, self.ids = base, ids
            def __len__(self):
                return len(self.ids)
            def __getitem__(self, i):
                return self.base[self.ids[i]]
        return _Sub(ds, ids)

    train_loader = DataLoader(subset(dataset, tr), batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=pad_collate, drop_last=False)
    valid_loader = DataLoader(subset(dataset, va if len(va)>0 else tr), batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=pad_collate, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoRankMLP(FEAT_DIM, d=160, p=0.1, num_classes=9).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs*len(train_loader)))

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, sched, device)
        va_acc = evaluate(model, valid_loader, device)
        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} valid_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, weights_out)
    print(f"Saved best weights to {weights_out} (valid_acc={best_acc:.4f})")


def run_infer(data_dir: str, weights: str, out_csv: str):
    device = torch.device("cpu")
    model = GoRankMLP(FEAT_DIM, d=160, p=0.0, num_classes=9).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    testset = GoRankDataset(data_dir, split="test")
    loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    rows: List[Tuple[str,int]] = []
    with torch.no_grad():
        for fids, x, lengths, _ in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            pred = (logits.argmax(dim=1) + 1).item()  # integer 1..9
            file_id = Path(fids[0]).stem  # like '1'
            rows.append((file_id, pred))

    # Write submission: id,label (numeric 1..9)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'label'])
        rows_sorted = sorted(rows, key=lambda r: int(r[0]))
        for fid, lab in rows_sorted:
            w.writerow([fid, lab])
    print(f"Saved submission to {out_csv} with {len(rows)} rows.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='./data_set', help='Root of dataset with train/ and test/')
    ap.add_argument('--out', type=str, default='submission_mlp.csv')
    ap.add_argument('--weights', type=str, default='weights_mlp.pt')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    if args.train or (not os.path.exists(args.weights)):
        print("[INFO] Training MLP (DeepSets)...")
        run_train(args.data_dir, args.weights, batch_size=args.batch_size, epochs=args.epochs)

    print("[INFO] Running inference and writing submission.csv ...")
    run_infer(args.data_dir, args.weights, args.out)
