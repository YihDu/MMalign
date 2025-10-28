import os, json, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.input, "*.json"))
    data = [json.load(open(f)) for f in json_files]
    df = pd.DataFrame(data)
    df["step"] = df["ckpt"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("step")

    df.to_csv(os.path.join(args.input, "geometry_summary.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["step"], df["rankme"], label="RankMe", marker="o")
    plt.plot(df["step"], df["alpha_req"], label="αReQ", marker="s")
    plt.xlabel("Training Step")
    plt.ylabel("Spectral Metric")
    plt.title("Representation Geometry Evolution (LLaVA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"[INFO] Plot saved to {args.output}")

if __name__ == "__main__":
    main()
