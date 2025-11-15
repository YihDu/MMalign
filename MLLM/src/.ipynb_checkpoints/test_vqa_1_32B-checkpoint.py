#!/usr/bin/env python

from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time
from tqdm import tqdm

import argparse
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time


class HiddenStateDataset(Dataset):
    def __init__(self, data_path, langs, max_samples=None):
        self.data_path = data_path
        self.langs = langs
        self.samples = self._load_samples(max_samples)

    def _load_samples(self, max_samples):
        samples = []
        print("Pre-loading sample metadata...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample_info = json.loads(line)
                for lang in self.langs:
                    if sample_info["questions"].get(lang):
                        samples.append({
                            "id": sample_info["id"],
                            "images": sample_info["images"][0],
                            "questions": sample_info["questions"][lang],
                            "options": sample_info.get("options", []),
                            "lang": lang
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_b64 = sample["images"]
        try:
            image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
            question = sample["questions"]
            options = sample["options"]
            text = f"{question}\n" + "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            )
            return {
                "qid": sample["id"],
                "lang": sample["lang"],
                "image": image,
                "text": text,
            }
        except Exception as e:
            print(f"Error loading sample {sample['id']}: {e}")
            return None 


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    qids = [item['qid'] for item in batch]
    langs = [item['lang'] for item in batch]
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    
    return {
        "qids": qids,
        "langs": langs,
        "images": images,
        "texts": texts
    }


def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    langs = args.langs.split(",")
    max_samples = args.max_samples
    batch_size = args.batch_size
    layer_interval = args.layer_interval
    cache_every = 8

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CONFIG ==========")
    print(f"langs={langs}, batch={batch_size}, layer_interval={layer_interval}, cache_every={cache_every}\n")

    # ---------------- Load model and processor ----------------
    print(f"🚀 Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": "cuda:1"},
)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    processor.tokenizer.padding_side = "left"
    print("✅ Model loaded and configured.\n")

    # ---------------- Load dataset ----------------
    dataset = HiddenStateDataset(data_path, langs, max_samples)

    # ============================================================
    # Process each language independently
    # ============================================================
    for lang in langs:
        skipped_batches = 0
        skipped_samples = 0
        print(f"\n===== Processing language: {lang} =====")
        lang_samples = [s for s in dataset.samples if s["lang"] == lang]
        lang_dataset = torch.utils.data.Subset(dataset, [dataset.samples.index(s) for s in lang_samples])
        lang_loader = DataLoader(
            lang_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )

        lang_save_dir = save_dir / lang
        lang_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Language {lang}: {len(lang_dataset)} samples, {len(lang_loader)} batches")
    

        # Thread-safe write pool per language
        pool = ThreadPoolExecutor(max_workers=4)
        pending, cache = [], []
        start_time = time.time()

        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(lang_loader, desc=f"{lang} batches")):
                if batch is None:
                    continue

                t0 = time.time()

                messages_list = [
                    [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img, },
                            {"type": "text", "text": txt},
                        ],
                    }]
                    for img, txt in zip(batch["images"], batch["texts"])
                ]

                text_inputs = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages_list
                ]

                image_inputs, video_inputs = process_vision_info(messages_list)
                inputs = processor(
                    text=text_inputs,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length= 8192,
                ).to("cuda:1", dtype=torch.float16)
        
                t1 = time.time()

                # ---------------- Forward pass ----------------
                # ---------------- Forward pass ----------------
                try:
                    outputs = model(**inputs, output_hidden_states=True)
                except ValueError as e:
                    err = str(e)
                    if "Image features and image tokens do not match" in err:
                        skipped_batches += 1
                        skipped_samples += len(batch["qids"])
                        print(f"[⚠️ Skip] Batch {batch_idx} ({len(batch['qids'])} samples) due to token mismatch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e  

                hidden_states = outputs.hidden_states
                t2 = time.time()

                # ---------------- Select layers ----------------
                selected_layers = [hidden_states[-1]] if layer_interval == 0 else hidden_states[::layer_interval]
                selected_layers = [h.to(torch.float16).cpu() for h in selected_layers]


                # ---------------- Compute per-sample mean ----------------
                for i, qid in enumerate(batch["qids"]):
                    hs_to_save = [layer_hs[i].mean(dim=0) for layer_hs in selected_layers]
                    if batch_idx == 0 and i < 3:
                        print(f"[Sample {qid}-{lang}] Layer0 mean={hs_to_save[0].mean():.4f}, std={hs_to_save[0].std():.4f}")
                    cache.append((qid, lang, hs_to_save))

                # ---------------- Async write ----------------
                if len(cache) >= cache_every * batch_size:
                    data_to_save = list(cache)
                    cache.clear()
                    pending.append(pool.submit(save_cache_batch, data_to_save, lang_save_dir))

                t3 = time.time()
                print(f"[Batch {batch_idx}] prep={t1-t0:.2f}s | forward={t2-t1:.2f}s | cache={t3-t2:.2f}s")

                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print(f"⚠️ Skipped {skipped_batches} batches, {skipped_samples} samples due to token mismatch.")

        
        # Save remaining cache
        if cache:
            pending.append(pool.submit(save_cache_batch, list(cache), lang_save_dir))
            cache.clear()

        for f in as_completed(pending):
            _ = f.result()
        pool.shutdown(wait=True)
        print(f"🎉 {lang} done in {time.time() - start_time:.2f}s")

def save_cache_batch(data_list, save_dir: Path):
    """批量异步写盘函数"""
    for qid, lang, hs in data_list:
        try:
            torch.save(hs, save_dir / f"sample_{qid}_{lang}.pt")
        except Exception as e:
            print(f"[⚠️] Failed to save {qid}_{lang}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract hidden states for multilingual MDUR samples using Qwen-VL-2.5."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, default="EN,ZH,AR",
                        help="Comma-separated list of languages.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Number of samples to process. Default: all.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference.")
    parser.add_argument("--layer_interval", type=int, default=1,
                        help="Save every N layers (e.g., 2→save layers 0,2,4,...; 0→only last layer).")

    args = parser.parse_args()
    main(args)