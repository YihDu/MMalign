import os
import re
import json
import numpy as np
from tqdm import tqdm
from utils.config_loader import load_config
from utils.spectral_metrics import (
    compute_rankme,
    compute_alpha_req,
    compute_effective_rank,
)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _parse_feature_filename(filename: str):
    """
    解析由 get_activation.py 产出的特征文件名，提取层号等信息。
    """
    match = re.match(r"^spectral_layer(\d+)_([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_(checkpoint-\d+)\.npy$", filename)
    if not match:
        return None
    layer_idx, token_scope, ckpt_tag = match.groups()
    return {
        "layer_idx": int(layer_idx),
        "token_scope": token_scope,
        "ckpt_tag": ckpt_tag,
    }


def compute_metric_from_activation(config_path: str):
    cfg = load_config(config_path)

    feature_dir = cfg.paths.activation_save_dir
    metric_dir = os.path.join(feature_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)

    requested_metrics = cfg.analysis.get("metrics", ["rankme", "alpha_req"])
    metric_fns = {
        "rankme": compute_rankme,
        "alpha_req": compute_alpha_req,
        "effective_rank": compute_effective_rank,
    }

    npy_files = [
        f for f in os.listdir(feature_dir)
        if f.endswith(".npy") and f.startswith("spectral_layer")
    ]
    results = []

    for f in tqdm(npy_files, desc="Computing spectral metrics"):
        meta = _parse_feature_filename(f)
        if meta is None:
            print(f"⚠️ 跳过无法解析的特征文件: {f}")
            continue

        step_name = f.replace(".npy", "")
        npy_path = os.path.join(feature_dir, f)
        out_json = os.path.join(metric_dir, f"{step_name}.json")
        if os.path.exists(out_json):
            print(f"Skip {step_name}, already exists.")
            continue

        features = np.load(npy_path).astype(np.float64)
        if features.ndim != 2:
            raise ValueError(f"期望二维特征矩阵，收到形状: {features.shape}")

        # 特征中心化，避免均值漂移影响协方差估计
        features -= features.mean(0, keepdims=True)
        cov = np.cov(features, rowvar=False)
        eigvals = np.sort(np.clip(np.linalg.eigvalsh(cov), a_min=0.0, a_max=None))[::-1]
        singular_vals = np.sqrt(np.maximum(eigvals, 0.0))

        metric_values = {}
        for name in requested_metrics:
            fn = metric_fns.get(name)
            if fn is None:
                print(f"⚠️ 指标 {name} 尚未实现，已跳过。")
                continue
            if name == "alpha_req":
                metric_values[name] = fn(eigvals)
            else:
                metric_values[name] = fn(singular_vals)

        hidden_dim = features.shape[1]

        result = {
            "ckpt": step_name,
            "metrics": metric_values,
            "num_samples": cfg.num_samples,
            "hidden_dim": hidden_dim,
            "layer": meta["layer_idx"],
            "token_scope": meta["token_scope"],
            "ckpt_tag": meta["ckpt_tag"],
        }
        # 向下兼容旧字段，方便已有可视化脚本读取
        for name, value in metric_values.items():
            result[name] = value

        save_json(result, out_json)
        results.append(result)
        # TODO: 根据指标类型自适应格式化输出
        metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metric_values.items())
        print(f"[DONE] {step_name}: {metric_str}, dim={hidden_dim}")

    summary_path = os.path.join(metric_dir, "summary.json")
    save_json(results, summary_path)
    print(f"✅ All done. Summary saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    compute_metric_from_activation(args.config)
