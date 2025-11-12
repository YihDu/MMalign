#!/usr/bin/env python

# 建议：考虑使用 ProcessPoolExecutor 以绕开 GIL
from concurrent.futures import ProcessPoolExecutor, as_completed 
import torch

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
from tqdm import tqdm


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
            return None # 返回 None，后续在 collate_fn 中处理


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

    # ===== 加载模型 =====
    print(f"🚀 Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    ).to('cuda')
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    processor.tokenizer.padding_side = "left"
    print("✅ Model loaded and configured.\n")
    
    dataset = HiddenStateDataset(data_path, langs, max_samples)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # 建议：可以根据CPU核心数和I/O性能调整 num_workers
        num_workers=8, 
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    print(f"Total batches to process: {len(data_loader)}\n")

    # 建议：使用 ProcessPoolExecutor 替代 ThreadPoolExecutor
    pool = ProcessPoolExecutor(max_workers=4) 
    pending, cache = [], []
    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            if batch is None:
                continue

            try:
                t0 = time.time()
                messages_list = [
                    [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": txt}
                    ]}]
                    for img, txt in zip(batch["images"], batch["texts"])
                ]
                text_inputs = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    for msg in messages_list
                ]
                image_inputs_list = [process_vision_info(msg)[0] for msg in messages_list]
                flat_image_inputs = [img for sub in image_inputs_list for img in sub]
                inputs = processor(
                    text=text_inputs,
                    images=flat_image_inputs,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                ).to('cuda') # 建议：将输入直接发送到 GPU
                t1 = time.time()

                # --- 推理 ---
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                t2 = time.time()

                # --- 层选择与数据准备 ---
                if layer_interval == 0:
                    selected_layers_gpu = [hidden_states[-1]]
                else:
                    selected_layers_gpu = hidden_states[::layer_interval]


                for i, (qid, lang) in enumerate(zip(batch["qids"], batch["langs"])):

                    hs_to_save_gpu = [layer_hs[i] for layer_hs in selected_layers_gpu]
                    cache.append((qid, lang, hs_to_save_gpu))
                t3 = time.time()

                if len(cache) >= cache_every * batch_size:
                    data_to_save = list(cache)
                    cache.clear()
                    pending.append(pool.submit(save_and_process_cache_batch, data_to_save, save_dir))

                print(f"[Batch {batch_idx}] prep={t1-t0:.2f}s | forward={t2-t1:.2f}s | cache={t3-t2:.2f}s")
                

            except Exception as e:
                print(f"[❌] Error in batch {batch_idx}: {e}")

    if cache:
        pending.append(pool.submit(save_and_process_cache_batch, list(cache), save_dir))
        cache.clear()

    for f in as_completed(pending):
        _ = f.result()
    pool.shutdown(wait=True)

    print(f"\n🎉 Done! Total time: {time.time() - start_time:.2f}s")


# ===========================
# 3️⃣ 优化的异步保存函数
# ===========================
def save_and_process_cache_batch(data_list, save_dir: Path):
    """在子进程中处理数据转换并批量异步写盘"""
    for qid, lang, hs_gpu in data_list:
        try:
            # 在这里进行 CPU 转换，充分利用子进程
            hs_cpu = [h.to(torch.float16).cpu() for h in hs_gpu]
            torch.save(hs_cpu, save_dir / f"sample_{qid}_{lang}.pt")
        except Exception as e:
            print(f"[⚠️] Failed to save {qid}_{lang}: {e}")

if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass  # 忽略 'context has already been set' 的错误

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