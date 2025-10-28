import os, json, argparse, numpy as np, torch
from utils.ckpt_loader import load_llava_ckpt
from analysis.extract_features import extract_features
from utils.spectral_metrics import compute_rankme, compute_alpha_req

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--layer", type=str, default="last")
    parser.add_argument("--token_scope", type=str, default="text_last")
    args = parser.parse_args()

    step_name = os.path.basename(args.ckpt.rstrip("/"))
    out_path = os.path.join(args.output, f"{step_name}.json")

    model = load_llava_ckpt(args.ckpt)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    features = extract_features(model, args.num_samples, args.layer, args.token_scope)
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    features -= features.mean(0, keepdims=True)
    cov = np.cov(features, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

    rankme = compute_rankme(eigvals)
    alpha = compute_alpha_req(eigvals)

    result = {"ckpt": step_name, "rankme": float(rankme), "alpha_req": float(alpha),
              "num_samples": args.num_samples, "layer": args.layer, "token_scope": args.token_scope}
    save_json(result, out_path)
    print(f"[DONE] {step_name}: RankMe={rankme:.3f}, αReQ={alpha:.3f}")

if __name__ == "__main__":
    main()
