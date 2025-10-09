#!/usr/bin/env python3
"""
Q5.py — Go Rank Prediction (LightGBM Baseline)

- Default behavior: trains on train_set and predicts test_set to produce submission.csv
- Training and inference are both contained here (single-file solution per spec)

Assumed directory layout after you unzip the Kaggle dataset:

./dataset/
    train_set/
        log_1D_policy_train.txt (muliple games inside)
        ...
        log_9D_policy_train.txt
    test_set/
        1.txt (single game)
        2.txt
        ...

Usage:
    python Q5_lightgbm.py --data_dir ./dataset --out submission.csv

Optional:
    python Q5.py --data_dir ./dataset --no-train  # (will try to load saved weights if available)

Notes:
- We aggregate move-level features to game-level when possible (detects "Game X:" headers).
- For test files, we aggregate the whole file to a single feature vector (the target is per file).
- Robust parser: ignores blank lines, tolerates variable spacing, and handles percentage signs.
- Features per move (31 dims):
    9 policy probs + 9 value preds + 9 rank-model probs + 1 strength + 3 KataGo (winrate, lead, uncertainty)
- Winrate & lead are given from BLACK perspective. For white moves, we convert to black-perspective:
    winrate := 1 - winrate, lead := -lead.

Author: b12902115
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import scipy.stats

import numpy as np

# LightGBM is allowed for Q5 per spec
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None
    print("[WARN] lightgbm not found. Please install with: pip install lightgbm", file=sys.stderr)


# ---------- Utilities ----------

RANKS = [f"{d}D" for d in range(1, 10)]  # '1'..'9'
RANK2IDX = {r: i for i, r in enumerate(RANKS)}
IDX2RANK = {i: r for r, i in RANK2IDX.items()}

MOVE_RE = re.compile(r"^[BW]\[[A-HJ-T](?:[1-9]|1[0-9])\]$")  # e.g., B[Q16]
GAME_HDR_RE = re.compile(r"^Game\s+\d+:", re.IGNORECASE)
PCT_RE = re.compile(r"%+$")  # strip trailing percent signs


def _to_floats(line: str) -> List[float]:
    parts = line.strip().split()
    out: List[float] = []
    for tok in parts:
        tok = PCT_RE.sub("", tok)  # remove % if any
        if tok == "-" or tok == "":
            continue
        try:
            out.append(float(tok))
        except ValueError:
            # Ignore non-numeric garbage gracefully
            pass
    return out


def _parse_one_move(block: List[str]) -> Optional[List[float]]:
    """Parse a 6-line move block into a 31-dim feature list.
       Returns None if the block is malformed.
    """
    if len(block) < 6:
        return None

    color_line = block[0].strip()
    if not MOVE_RE.match(color_line):
        return None
    is_white = color_line.startswith("W[")

    # 1. Policy: 9 floats on its line
    policy = _to_floats(block[1])
    total = sum(policy)
    policy = [p / total for p in policy] if sum(policy) > 0 else [1.0/9]*9
    policy_max = np.argmax(policy)
    policy_pos_diff = policy_max - np.argsort(policy)[-2]

    def entropy(prob: List[float], eps: float = 1e-8) -> float:
        prob = np.asarray(prob, dtype=float)  # 轉成 numpy 陣列
        sum_p = np.sum(prob) + eps
        prob_norm = prob / sum_p

        prob_sorted = np.sort(prob_norm)
        p1 = prob_sorted[-1]
        p2 = prob_sorted[-2]
        prob_margin = p1 - p2

        entropy_val = -np.sum(prob_norm * np.log(prob_norm + eps))
        entropy_max = np.log(len(prob_norm))  # log(9)
        entropy_compact = 1.0 - (entropy_val / (entropy_max + eps))  # 歸一化 (0~1)

        margin_weighted_ent = prob_margin * entropy_compact

        return float(margin_weighted_ent)
    
    policy_entropy = entropy(policy)
    policy_times_ent = [p * policy_entropy for p in policy]

    # 2. Value Predictions: 9 floats on its line
    values = _to_floats(block[2])
    values_max = np.argmax(values)
    values_pos_diff = values_max - np.argsort(values)[-2]

    # 3. Rank Model Outputs: 9 floats on its line
    rankouts = _to_floats(block[3])
    rankouts_max = np.argmax(rankouts)
    rankouts_pos_diff = rankouts_max - np.argsort(rankouts)[-2]
    rankouts_weighted_avg = np.average(rankouts, weights=np.arange(1, 10))

    rankouts_entropy = entropy(rankouts)
    rankouts_times_ent = [r * rankouts_entropy for r in rankouts]

    # 4. Strength: single float on its line
    strength_list = _to_floats(block[4])
    strength = strength_list[0] 

    # 5. KataGo: 3 floats (winrate%, lead, uncertainty) on its line
    kata = _to_floats(block[5])
    kata_wr, kata_lead, kata_unc = kata[0], kata[1], kata[2]
    kata_wr /= 100.0
    if is_white:
        kata_wr = 1.0 - kata_wr
        kata_lead = -kata_lead
    kata_wr_err = abs(kata_wr - 0.5)
    kata_lead_abs = abs(kata_lead)

    wr_diff = [abs(values[i] - kata_wr) for i in range(9)]

    feats = []
    feats.extend(policy)        # 9
    feats.append(policy_max)      # +1 
    feats.append(policy_pos_diff)      # +1 
    feats.append(policy_entropy)  # +1 
    feats.extend(policy_times_ent)  # +9 

    feats.extend(values)        # +9 
    feats.append(values_max)      # +1 
    feats.append(values_pos_diff)      # +1 

    feats.extend(rankouts)      # +9 
    feats.append(rankouts_max)    # +1 
    feats.append(rankouts_pos_diff)    # +1 
    feats.append(rankouts_weighted_avg)  # +1 
    feats.append(rankouts_entropy)  # +1
    feats.extend(rankouts_times_ent)  # +9

    feats.append(strength)        # +1 

    feats.append(kata_wr)         # +1 
    feats.append(kata_wr_err)     # +1
    feats.append(kata_lead)       # +1
    feats.append(kata_lead_abs)   # +1
    feats.append(kata_unc)        # +1 

    feats.extend(wr_diff)        # +9 = 69
    return feats


def parse_moves_from_file(fp: Path) -> List[List[float]]:
    """Parse all move blocks (6 lines each) from a file.
       Ignores any non-move lines. Returns list of 31-dim feature lists.
    """
    lines = [ln.rstrip("\n") for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines()]
    # Keep only meaningful lines and mark game headers if present
    cleaned: List[str] = [ln.strip() for ln in lines if ln.strip()]

    moves: List[List[float]] = []
    i = 0
    N = len(cleaned)
    while i < N:
        if GAME_HDR_RE.match(cleaned[i]):
            i += 1
            continue
        if MOVE_RE.match(cleaned[i]) and i + 5 < N:
            block = cleaned[i:i+6]
            feats = _parse_one_move(block)
            if feats is not None:
                moves.append(feats)
                i += 6
                continue
        # If not a valid block, skip one line and keep scanning
        i += 1
    return moves


def aggregate_features(moves: List[List[float]]) -> np.ndarray:
    """Aggregate move-level features to a fixed-length vector per (game/file).
       We use simple stats: mean, std, min, max for each dim + move_count.
       Output shape: 69*18 + 1 = 901 dims.
    """
    if not moves:
        # Return zeros if empty (shouldn't happen if data is well-formed)
        return np.zeros(69 * 18 + 1, dtype=np.float32)
    X = np.asarray(moves, dtype=np.float32)  
    n_moves, n_feats = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    vmin = X.min(axis=0)
    vmax = X.max(axis=0)
    q20 = np.percentile(X, 20, axis=0)
    q80 = np.percentile(X, 80, axis=0)
    median = np.median(X, axis=0)

    CUTOFF1 = 50
    CUTOFF2 = 150

    # 2. Early Game / Fuseki
    early = X[:CUTOFF1, :]
    if early.shape[0] > 0:
        early_mean = early.mean(axis=0)
        early_std = early.std(axis=0)
    else:
        early_mean = np.zeros(n_feats)
        early_std = np.zeros(n_feats)

    # 3. Mid Game / Chuban
    mid = X[CUTOFF1:CUTOFF2, :]
    if mid.shape[0] > 0:
        mid_mean = mid.mean(axis=0)
        mid_std = mid.std(axis=0)
        mid_median = np.median(mid, axis=0)
        mid_q20 = np.percentile(mid, 20, axis=0)
        mid_q80 = np.percentile(mid, 80, axis=0)

        # safe compute skew and kurtosis to avoid precision loss warnings
        mid_skew = np.zeros(n_feats)
        mid_kurt = np.zeros(n_feats)
        for i in range(n_feats):
            col_data = mid[:, i]
            if len(np.unique(col_data)) > 2 and np.std(col_data) > 1e-8:
                mid_skew[i] = scipy.stats.skew(col_data)
                mid_kurt[i] = scipy.stats.kurtosis(col_data)
    else:
        mid_mean = np.zeros(n_feats)
        mid_std = np.zeros(n_feats)
        mid_median = np.zeros(n_feats)
        mid_q20 = np.zeros(n_feats)
        mid_q80 = np.zeros(n_feats)

        mid_skew = np.zeros(n_feats)
        mid_kurt = np.zeros(n_feats)

    # 4. Late Game / Yose
    late = X[CUTOFF2:, :]
    if late.shape[0] > 0:
        late_mean = late.mean(axis=0)
        late_std = late.std(axis=0)
    else:
        late_mean = np.zeros(n_feats)
        late_std = np.zeros(n_feats)

    #  print(f"[DEBUG] early_mean={early_mean}, mid_mean={mid_mean}, late_mean={late_mean}")

    out = np.concatenate([mean, std, median, vmin, vmax, q20, q80]) # 7
    out = np.concatenate([out, early_mean, mid_mean, late_mean, early_std, mid_std, late_std, mid_median, mid_q20, mid_q80, mid_skew, mid_kurt]) # +11 = 18
    #print(f"[DEBUG] Aggregated {n_moves} moves -> feature: {out}")
    out = np.concatenate([out, np.array([float(len(moves))])]) # +1
    return out.astype(np.float32)


def split_games(moves: List[List[float]], raw_lines: List[str]) -> List[List[List[float]]]:
    """Split moves into games if the file contains 'Game X:' headers.
       Fallback: if no headers, return a single game with all moves.
    """
    # Simple detection: if any 'Game X:' lines exist in raw text, split by them
    game_indices: List[int] = []
    for idx, ln in enumerate(raw_lines):
        if GAME_HDR_RE.match(ln.strip()):
            game_indices.append(idx)
    if not game_indices:
        return [moves]

    # Re-parse but collecting per-game moves using header positions.
    # We do a second pass so that we can align moves roughly to game segments.
    games: List[List[List[float]]] = []
    current: List[List[float]] = []
    i = 0
    cleaned = [ln.strip() for ln in raw_lines if ln.strip()]
    N = len(cleaned)
    while i < N:
        if GAME_HDR_RE.match(cleaned[i]):
            if current:
                games.append(current)
                current = []
            i += 1
            continue
        if MOVE_RE.match(cleaned[i]) and i + 5 < N:
            feats = _parse_one_move(cleaned[i:i+6])
            if feats is not None:
                current.append(feats)
                i += 6
                continue
        i += 1
    if current:
        games.append(current)
    if not games:
        games = [moves]
    return games


def parse_file_aggregate(fp: Path) -> Tuple[np.ndarray, int]:
    """Parse a training file and return stacked game-level features (many rows) and label idx.
       Label is inferred from filename like 'log_7D_policy_train.txt' -> '7D'.
    """
    text = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    # First pass: collect all moves (for fallback and for splitting)
    all_moves = parse_moves_from_file(fp)
    games = split_games(all_moves, text)
    feats = [aggregate_features(gm) for gm in games if gm]

    # Infer label from filename
    m = re.search(r"log_(\dD)_policy_train\\.txt$", fp.name)
    if not m:
        # fallback: try any *D pattern
        m = re.search(r"(\dD)", fp.name)
    if not m:
        raise ValueError(f"Cannot infer rank label from filename: {fp}")
    rank = m.group(1)
    y_idx = RANK2IDX[rank]

    X = np.vstack(feats) if feats else np.zeros((0, 69 * 18 + 1), dtype=np.float32)
    y = np.full((X.shape[0],), y_idx, dtype=np.int64)
    return X, y_idx, y


def load_train_set(train_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for d in range(1, 10):
        fp = train_dir / f"log_{d}D_policy_train.txt"
        if not fp.exists():
            print(f"[WARN] Missing {fp}")
            continue
        Xd, y_idx, yd = parse_file_aggregate(fp)
        if Xd.shape[0] == 0:
            print(f"[WARN] No samples parsed from {fp}")
            continue
        X_list.append(Xd)
        y_list.append(yd)
        print(f"[INFO] Parsed {fp.name}: {Xd.shape[0]} games -> label {IDX2RANK[y_idx]}")
    X = np.vstack(X_list) if X_list else np.zeros((0, 69 * 18 + 1), dtype=np.float32)
    y = np.concatenate(y_list) if y_list else np.zeros((0,), dtype=np.int64)
    print(f"[INFO] Train set: X={X.shape}, y={y.shape}")
    return X, y


def load_test_set(test_dir: Path) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    feats: List[np.ndarray] = []
    for fp in sorted(test_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        moves = parse_moves_from_file(fp)
        Xvec = aggregate_features(moves)
        feats.append(Xvec)
        ids.append(fp.stem)
        print(f"[INFO] Parsed test file {fp.name}: moves={len(moves)} -> features shape={Xvec.shape}")
    X = np.vstack(feats) if feats else np.zeros((0, 51 * 18 + 1), dtype=np.float32)
    return ids, X

# adjust hyperparameters here
def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if lgb is None:
        raise ImportError("lightgbm is not installed. Please pip install lightgbm")
    params = dict(
        objective="multiclass",
        num_class=9,
        learning_rate=0.05, # 0.01 - 0.05
        n_estimators=1500, # - 2000
        subsample=0.8, # 0.8 - 1.0
        colsample_bytree=0.9,
        num_leaves=255,
        max_depth=-1,
        random_state=seed,
        n_jobs=-1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def predict_rank(model, X: np.ndarray) -> List[str]:
    proba = model.predict_proba(X)  # list of arrays per class or ndarray shape [n, 9]
    if isinstance(proba, list):
        proba = np.stack(proba, axis=1)
    idx = np.argmax(proba, axis=1)
    return [IDX2RANK[i] for i in idx]


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.booster_.save_model(str(path))
        print(f"[INFO] Saved LightGBM model to {path}")
    except Exception:
        # Fallback to pickle
        import pickle
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] Saved model pickle to {path.with_suffix('.pkl')}")


def load_model(path: Path):
    if not path.exists() and not path.with_suffix('.pkl').exists():
        return None
    try:
        booster = lgb.Booster(model_file=str(path))
        # Wrap booster in an LGBMClassifier-like object
        clf = lgb.LGBMClassifier()
        clf._Booster = booster
        clf.fitted_ = True
        return clf
    except Exception:
        import pickle
        with open(path.with_suffix('.pkl'), 'rb') as f:
            return pickle.load(f)


def write_submission(ids: List[str], ranks: List[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('id,label\n')
        for i, r in zip(ids, ranks):
            label = RANK2IDX[r] + 1
            f.write(f"{i},{label}\n")
    print(f"[INFO] Wrote submission to {out_csv}")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='./data_set', help='Root folder containing train_set/ and test_set/')
    ap.add_argument('--out', type=str, default='submission.csv', help='Output CSV path')
    ap.add_argument('--weights', type=str, default='./weights/lgb_model.txt', help='Model weight path (LightGBM text format or pickle)')
    ap.add_argument('--no-train', action='store_true', help='Skip training and load weights if available')
    ap.add_argument('--no-test', action='store_true', help='Skip test prediction (for debugging)')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio (0 to disable split)')
    ap.add_argument('--train-all-after-val', action='store_true',
                    help='After validation/CV, retrain on FULL train_set before predicting test_set.')
    
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train_set'
    test_dir = data_dir / 'test_set'

    # Load data & train
    if not args.no_train:
        X, y = load_train_set(train_dir)
        if X.shape[0] == 0:
            print('[ERROR] No training samples parsed. Check dataset path or parser assumptions.', file=sys.stderr)
            sys.exit(1)

        if args.train_all_after_val:
            print("[INFO] Retraining on FULL train_set after validation...")
            model = train_lightgbm(X, y, seed=args.seed)
        
        elif args.val_ratio and 0.0 < args.val_ratio < 1.0:
            # Split train/val 
            print(f"[INFO] Splitting train/val with ratio {args.val_ratio}...")
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=args.val_ratio, stratify=y, random_state=args.seed
            )

            model = train_lightgbm(Xtr, ytr, seed=args.seed)

            # Validate
            yhat = model.predict(Xva)
            acc = accuracy_score(yva, yhat)
            print(f"[VALID] Accuracy = {acc:.4f}")

        # Save model
        save_model(model, Path(args.weights))
    else:
        model = load_model(Path(args.weights))
        if model is None:
            print('[ERROR] --no-train specified but weights not found.', file=sys.stderr)
            sys.exit(1)

    # Predict test
    if not args.no_test:
        ids, Xte = load_test_set(test_dir)
        if Xte.shape[0] == 0:
            print('[ERROR] No test samples found.', file=sys.stderr)
            sys.exit(1)
        ranks = predict_rank(model, Xte)
        write_submission(ids, ranks, Path(args.out))


if __name__ == '__main__':
    main()
