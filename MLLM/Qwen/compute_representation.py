#!/usr/bin/env python3

import torch
import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import argparse



def mean_center(X: torch.Tensor):
    return X - X.mean(0, keepdim=True)

def normalize(X: torch.Tensor):
    return X / (X.norm(dim=-1, keepdim=True) + 1e-8)

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.flatten().float()
    Y = Y.flatten().float()
    X -= X.mean()
    Y -= Y.mean()
    num = (X @ Y) ** 2
    denom = (X @ X) * (Y @ Y)
    return (num / denom).item()

def cosine_tokenwise(X, Y):
    X, Y = ensure_2d(X), ensure_2d(Y)
    Xn, Yn = normalize(X), normalize(Y)
    return (Xn * Yn).sum().item()  # 单向量 dot product

def ensure_2d(X):
    return X.unsqueeze(0) if X.dim() == 1 else X

def normalized_cosine(X, Y, eps=1e-6):
    X, Y = ensure_2d(X), ensure_2d(Y)
    cos_xy = cosine_tokenwise(X, Y)
    cos_xx = cosine_tokenwise(X, X)
    cos_yy = cosine_tokenwise(Y, Y)
    cosl1 = min(cos_xy / (cos_xx + eps), 1.0)
    cosl2 = min(cos_xy / (cos_yy + eps), 1.0)
    return 2 * cosl1 * cosl2 / (cosl1 + cosl2 + eps)

def load_hidden_states(data_dir: Path, langs):
    grouped = {}
    for lang in langs:
        lang_dir = data_dir / lang
        if not lang_dir.exists():
            print(f"[⚠️] Warning: {lang_dir} not found, skipping.")
            continue
        for f in lang_dir.glob("sample_*.pt"):
            sid = f.stem.split("_")[1]
            grouped.setdefault(sid, {})[lang] = torch.load(f, map_location="cpu")
    return grouped


def compute_alignment(grouped, langs, metric="cka"):
    results = []
    pairs = list(combinations(langs, 2))
    print(f"Found {len(grouped)} samples; computing {metric.upper()} across {pairs} pairs")

    for (l1, l2) in pairs:
        for sid, sample in tqdm(grouped.items(), desc=f"{l1}-{l2}"):
            if l1 not in sample or l2 not in sample:
                continue
            layers = min(len(sample[l1]), len(sample[l2]))
            for layer_idx in range(layers):
                X, Y = sample[l1][layer_idx], sample[l2][layer_idx]
                if metric == "cka":
                    score = linear_cka(X, Y)
                elif metric == "cosine":
                    score = cosine_tokenwise(X, Y)
                elif metric == "cosine_norm":
                    score = normalized_cosine(X, Y)
                else:
                    raise ValueError(f"输入 正确 metric")
                results.append({
                    "sample": sid,
                    "layer": layer_idx,
                    "lang1": l1,
                    "lang2": l2,
                    "metric": metric,
                    "score": score
                })
    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, default="EN,ZH,AR")
    parser.add_argument("--metric", type=str, default="cka", choices=["cka", "cosine", "cosine_norm"])
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    langs = args.langs.split(",")
    grouped = load_hidden_states(Path(args.data_dir), langs)
    df = compute_alignment(grouped, langs, args.metric)
    df.to_csv(args.save_path, index=False)
    print(f"✅ Saved results to {args.save_path}")


if __name__ == "__main__":
    main()
