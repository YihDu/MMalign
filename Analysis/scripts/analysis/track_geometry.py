import os, json, numpy as np
from tqdm import tqdm
from utils.config_loader import load_config
from utils.spectral_metrics import compute_rankme, compute_alpha_req

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def compute_metric_from_activation(config_path: str):
    cfg = load_config(config_path)

    feature_dir = cfg.paths.activation_save_dir
    metric_dir = os.path.join(feature_dir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(feature_dir) if f.endswith("_features.npy")]
    results = []

    for f in tqdm(npy_files, desc="Computing spectral metrics"):
        step_name = f.replace("_features.npy", "")
        npy_path = os.path.join(feature_dir, f)
        out_json = os.path.join(metric_dir, f"{step_name}.json")
        if os.path.exists(out_json):
            print(f"Skip {step_name}, already exists.")
            continue

        features = np.load(npy_path)
        features -= features.mean(0, keepdims=True)
        cov = np.cov(features, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

        rankme = compute_rankme(eigvals)
        alpha = compute_alpha_req(eigvals)
        hidden_dim = features.shape[1]

        result = {
            "ckpt": step_name,
            "rankme": float(rankme),
            "alpha_req": float(alpha),
            "num_samples": cfg.num_samples,
            "hidden_dim": hidden_dim,
            "layer": cfg.model.target_layer,
            "token_scope": cfg.model.token_scope
        }

        save_json(result, out_json)
        results.append(result)
        print(f"[DONE] {step_name}: RankMe={rankme:.3f}, αReQ={alpha:.3f}, dim={hidden_dim}")

    summary_path = os.path.join(metric_dir, "summary.json")
    save_json(results, summary_path)
    print(f"✅ All done. Summary saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    compute_metric_from_activation(args.config)
