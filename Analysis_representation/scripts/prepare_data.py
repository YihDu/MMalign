"""Quick CLI helper to sanity-check the parquet COCO multilingual dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.coco_loader import COCODataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing parquet shards.")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="One or more dataset splits to sample from.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        required=True,
        help="Languages that must be present in the preview batch.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of samples to preview.")
    parser.add_argument("--caption-index", type=int, default=0, help="Preferred caption index per language.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow samples with missing/empty captions (they will be skipped otherwise).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = COCODataset(
        data_dir=args.data_dir,
        splits=args.splits,
        seed=args.seed,
        caption_index=args.caption_index,
        filter_empty_languages=not args.allow_empty,
    )
    batch = dataset.build_batch(limit=args.limit, languages=args.languages)
    print(f"Prepared {len(batch.examples)} examples covering languages: {', '.join(args.languages)}")
    for idx, example in enumerate(batch.examples, start=1):
        header = f"[{idx}] cocoid={example.image.image_id}"
        if example.image.filename:
            header += f" filename={example.image.filename}"
        print(header)
        for lang in args.languages:
            caption = example.captions[lang]
            preview = caption.text.replace("\n", " ")
            print(f"  - {lang}: {preview}")


if __name__ == "__main__":
    main()
