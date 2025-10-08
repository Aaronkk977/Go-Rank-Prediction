# =============================
# File: utils.py
# =============================
import os
import re
import numpy as np
import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

RANK_TO_ID = {f"{i}D": i-1 for i in range(1, 10)}  # '1D'->0 ... '9D'->8
ID_TO_RANK = {v: k for k, v in RANK_TO_ID.items()}

MOVE_RE = re.compile(r"^[BW]\[[A-T][0-9]{1,2}\]")

# === Feature extraction ===
# Each move has 6 lines in the training logs according to spec.
# 1: Move (e.g., B[Q16])
# 2: Policy(9)
# 3: Value(9)
# 4: Rank model outputs(9)
# 5: Strength(1)
# 6: KataGo: winrate, lead, uncertainty (3)
# NOTE: If the dataset you parse provides additional engineered feats, update INPUT_DIM accordingly.

INPUT_DIM = 31  # 9 + 9 + 9 + 1 + 3


def _to_floats(line: str) -> List[float]:
    # support percentages like "50.146%" -> 0.50146, and plain floats
    arr = []
    for tok in line.strip().split():
        if tok.endswith('%'):
            tok = tok[:-1]
            try:
                arr.append(float(tok) / 100.0)
            except ValueError:
                pass
        else:
            try:
                arr.append(float(tok))
            except ValueError:
                pass
    return arr


def parse_moves_from_lines(lines: List[str]) -> List[List[float]]:
    """
    Parse a single file's raw lines into a list of per-move feature vectors (length = INPUT_DIM).
    Robust to minor format inconsistencies; skips malformed blocks.
    """
    feats: List[List[float]] = []
    block: List[str] = []

    for ln in lines:
        if MOVE_RE.match(ln.strip()):
            # flush previous block if complete
            if len(block) == 6:
                f = _parse_one_move(block)
                if f is not None:
                    feats.append(f)
            block = [ln]
        else:
            if ln.strip():
                block.append(ln)

    # tail
    if len(block) == 6:
        f = _parse_one_move(block)
        if f is not None:
            feats.append(f)

    return feats


def _parse_one_move(block: List[str]) -> List[float]:
    if len(block) < 6:
        return None
    # block[0]: move string (ignored except color inversion for KataGo)
    move = block[0].strip()
    is_white = move.startswith('W[')

    policy = _to_floats(block[1])
    values = _to_floats(block[2])
    rankouts = _to_floats(block[3])
    strength = _to_floats(block[4])
    katago = _to_floats(block[5])  # [winrate(black), lead(black), uncertainty]

    if len(policy) != 9 or len(values) != 9 or len(rankouts) != 9:
        return None
    if len(strength) < 1 or len(katago) < 3:
        return None

    winrate_b, lead_b, uncert = katago[:3]
    # Invert perspective for white moves as per spec (winrate/lead are black's perspective)
    if is_white:
        winrate_b = 1.0 - winrate_b
        lead_b = -lead_b

    vec = policy + values + rankouts + [strength[0], winrate_b, lead_b, uncert]
    if len(vec) != INPUT_DIM:
        return None
    return vec


class GoSequenceDataset(Dataset):
    def __init__(self, samples: List[Dict], mean=None, std=None, clamp_std=1e-6, max_len: int = 512):
        """
        samples: list of { 'seq': np.ndarray [T,F], 'y': int }
        Performs z-score normalization using provided mean/std or computes from data.
        """
        self.max_len = max_len
        X_all = [s['seq'] for s in samples]
        if mean is None or std is None:
            concat = np.concatenate(X_all, axis=0) if len(X_all) > 0 else np.zeros((1, INPUT_DIM))
            self.mean = concat.mean(axis=0, keepdims=True)
            self.std = concat.std(axis=0, keepdims=True)
        else:
            self.mean = mean
            self.std = std
        self.std = np.maximum(self.std, clamp_std)

        self.samples = []
        for s in samples:
            seq = (s['seq'] - self.mean) / self.std
            # clip / pad sequence to max_len
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]
            self.samples.append({
                'seq': seq.astype(np.float32),
                'y': s.get('y', -1)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return torch.from_numpy(item['seq']), item['y']


def collate_batch(batch):
    # batch: list of (seq_tensor[T,F], y)
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    F = seqs[0].shape[1]
    max_len = max([s.shape[0] for s in seqs])
    padded = torch.zeros((len(seqs), max_len, F), dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    ys = torch.tensor(ys, dtype=torch.long) if ys[0] != -1 else torch.full((len(seqs),), -1, dtype=torch.long)
    return padded, lengths, ys


def read_train_split(train_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Reads 9 rank files from train_dir. Creates a simple train/val split (e.g., last 10% as val per file).
    Adjust the splitting rule to your preference if needed.
    Returns: train_samples, val_samples
    """
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []

    for rank in range(1, 10):
        fname = os.path.join(train_dir, f"log_{rank}D_policy_train.txt")
        if not os.path.isfile(fname):
            print(f"[WARN] Missing {fname}")
            continue
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        feats = parse_moves_from_lines(lines)
        if len(feats) == 0:
            continue

        seqs = []
        current_game = []

        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped.startswith("Game "):  # e.g. "Game 1:"
                    # 如果前一局有累積內容，先加入
                    if current_game:
                        moves = parse_moves_from_lines(current_game)
                        if len(moves) >= 32:  # 避免太短的無效局
                            seqs.append(np.asarray(moves, dtype=np.float32))
                        current_game = []
                else:
                    current_game.append(line)
            # flush 最後一局
            if current_game:
                moves = parse_moves_from_lines(current_game)
                if len(moves) >= 32:
                    seqs.append(np.asarray(moves, dtype=np.float32))
                
        # split
        n = len(seqs)
        n_val = max(1, int(0.1 * n))
        val_idx = set(range(n - n_val, n))
        for i, seq in enumerate(seqs):
            rec = {'seq': seq, 'y': rank - 1}
            if i in val_idx:
                val_samples.append(rec)
            else:
                train_samples.append(rec)
        
        print(f"[INFO] Parsed rank {rank}D: {len(seqs)} samples ({n - n_val} train, {n_val} val)")
    return train_samples, val_samples


def read_test_files(test_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Reads every *.txt in test_dir and returns list of (file_id, seq_array[T,F])."""
    out = []
    for fn in sorted(os.listdir(test_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if not fn.endswith('.txt'):
            continue
        file_id = os.path.splitext(fn)[0]
        path = os.path.join(test_dir, fn)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        feats = parse_moves_from_lines(lines)
        if len(feats) == 0:
            feats = [np.zeros(INPUT_DIM, dtype=np.float32)]  # avoid empty
        seq = np.asarray(feats, dtype=np.float32)
        out.append((file_id, seq))
    return out