#!/usr/bin/env python

# 修改 文本token

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import torch
log_file = open("log(Qwen_7B_1113).log", "a")

def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()

class PM4BenchVQA(Dataset):
    def __init__(self, data_path, langs, max_samples=None):
        self.langs = langs
        self.samples = self._load_samples(data_path, max_samples)

    def _load_samples(self, data_path, max_samples):
        samples = []
        print("Pre-loading sample metadata...")

        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                s = json.loads(line)

                # decode all images only once
                pil_images = [
                    Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    for b64 in s["images"]
                ]

                # use the first image as the main visual input (same as你的代码)
                main_img = pil_images[0]

                # build multi-language sub-samples
                for lang in self.langs:
                    q = s["questions"].get(lang)
                    opts = s["options"].get(lang)

                    if q is None or opts is None:
                        continue

                    text = q + "\n" + "\n".join(
                        [f"{chr(65+i)}. {o}" for i, o in enumerate(opts)]
                    )

                    samples.append({
                        "qid": s["id"],
                        "lang": lang,
                        "image": main_img,  
                        "text": text,
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
    layer_interval = args.layer_interval
    cache_every = 8

    batch_size = len(langs)  # ★ 多语言对齐的批大小 = 语言数

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CONFIG ==========")
    print(f"langs={langs}, batch={batch_size}, layer_interval={layer_interval}, cache_every={cache_every}\n")

    print(f"Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype='auto',
        attn_implementation="flash_attention_2",
        device_map="cuda",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    processor.tokenizer.padding_side = "left"
    print("Model loaded!\n")

    # ---------------- Load dataset ----------------
    dataset = PM4BenchVQA(data_path, langs, max_samples)

    # ============================================================
    # ★ Step 1：按 qid 将样本聚合成 batch
    # dataset.samples 格式是 [12 EN, 12 ZH, 12 AR, 13 EN, 13 ZH, 13 AR ...]
    # ============================================================
    from collections import defaultdict
    grouped = defaultdict(list)

    for idx, s in enumerate(dataset.samples):
        grouped[s["qid"]].append(idx)

    # 现在每个 grouped[qid] = [idx_EN, idx_ZH, idx_AR]
    batches = list(grouped.values())

    print(f"\nTotal qids = {len(batches)}, Will run {len(batches)} batches.")

    # ----------------------------------------------------------
    # ★ Step 2：构建一个人工 DataLoader（因为我们已有 batch indices）
    # ----------------------------------------------------------
    class ManualBatchLoader:
        def __init__(self, dataset, batches):
            self.dataset = dataset
            self.batches = batches

        def __len__(self):
            return len(self.batches)

        def __iter__(self):
            for b in self.batches:
                batch = [self.dataset[i] for i in b]
                yield custom_collate_fn(batch)

    loader = ManualBatchLoader(dataset, batches)

    # ---------------- Prepare save dirs ----------------
    lang_save_dirs = {}
    for lang in langs:
        d = save_dir / lang
        d.mkdir(parents=True, exist_ok=True)
        lang_save_dirs[lang] = d

    # Thread pool for async save
    pool = ThreadPoolExecutor(max_workers=4)
    pending = []
    cache = []
    start_time = time.time()

    # ==========================================================
    # ★ Step 3：真实多语言对齐 batch 推理
    # ==========================================================
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Multilingual Batches")):
            if batch is None:
                continue

            # --------- Debug print ---------
            if batch_idx < 2:
                print("\n====== DEBUG BATCH CHECK ======")
                print(f"Batch {batch_idx}")
                print(f"QIDs   : {batch['qids']}")
                print(f"LANGs  : {batch['langs']}")
                print(f"Unique langs: {set(batch['langs'])}")

                img0 = batch["images"][0]
                same_img = all(img0.size == im.size for im in batch["images"])
                print(f"All images same size: {same_img}")

                for bi in range(min(2, len(batch['texts']))):
                    preview = batch['texts'][bi][:120].replace("\n", "\\n")
                    print(f"Sample {bi}: lang={batch['langs'][bi]}, text={preview}...")
                print("================================\n")

            log(f"\n\n===== Batch {batch_idx} =====")
            log(f"QIDs: {batch['qids']}")
            log(f"LANGs: {batch['langs']}")

            # 文本长度
            log("text lens: " + str([len(t) for t in batch['texts']]))

            # 图片尺寸
            log("image sizes: " + str([im.size for im in batch['images']]))
            
            # ---------------- build messages ----------------
            t0 = time.time()
            messages_list = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
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
                max_length=8192,
            ).to("cuda")
            
            # vision special tokens
            vstart = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vend = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            # ===== DEBUG: Check input shapes / NaN / vision tokens =====
            log(f"input_ids shape: {inputs['input_ids'].shape}")
            log(f"attention_mask shape: {inputs['attention_mask'].shape}")



            vstart_count = (inputs["input_ids"] == vstart).sum().item()
            vend_count = (inputs["input_ids"] == vend).sum().item()
            log(f"vision_start count = {vstart_count}, vision_end count = {vend_count}")

            # check NaN
            has_nan = torch.isnan(inputs["input_ids"].float()).any().item()
            log(f"NaN in input_ids: {has_nan}")

            # check sequence max length for debugging
            seq_len = inputs["input_ids"].shape[1]
            log(f"Sequence length = {seq_len}")

            t1 = time.time()

            # ---------------- Forward pass ----------------
            try:
                outputs = model(**inputs, output_hidden_states=True)

            except Exception as e:  # 一定要捕获所有异常，包括 CUDA RuntimeError
                err = str(e)

                log(f"[ERROR] Batch {batch_idx} crash!")
                log(f"Exception: {repr(e)}")

                # 打印前 50 个 token
                try:
                    log("First 50 input_ids: " + str(inputs["input_ids"][0][:50].tolist()))
                except Exception:
                    log("Failed to print input_ids preview.")

                # 打印 vision token 在序列中位置
                try:
                    pos = [
                        i for i, x in enumerate(inputs["input_ids"][0].tolist())
                        if x == vstart or x == vend
                    ]
                    log("vision token positions: " + str(pos[:20]))
                except Exception:
                    log("Failed to locate vision tokens.")

                # 打印对应文本（非常重要）
                try:
                    kill_text = batch['texts'][0].replace("\n", "\\n")[:300]
                    log("Sample text preview: " + kill_text)
                except Exception:
                    log("Failed to print sample text preview.")

                # 专门处理 vision mismatch
                if "Image features and image tokens do not match" in err:
                    print(f"[⚠️ Skip] Batch {batch_idx} token mismatch.")
                    torch.cuda.empty_cache()
                    continue

                # 其它异常必须重新抛出
                raise e

            hidden_states = outputs.hidden_states
            t2 = time.time()

            # ---------------- Select layers ----------------
            selected_layers = (
                [hidden_states[-1]]
                if layer_interval == 0
                else hidden_states[::layer_interval]
            )
            selected_layers = [h.to(torch.float16).cpu() for h in selected_layers]

            # ---------------- Cache per-sample results ----------------
            # for i, (qid, lang) in enumerate(zip(batch["qids"], batch["langs"])):
            #     hs_to_save = [layer_hs[i].mean(dim=0) for layer_hs in selected_layers]

            #     if batch_idx == 0 and i < 3:
            #         print(f"[Sample {qid}-{lang}] mean={hs_to_save[0].mean():.4f}, std={hs_to_save[0].std():.4f}")

            #     cache.append((qid, lang, hs_to_save))
            
            for i, (qid, lang) in enumerate(zip(batch["qids"], batch["langs"])):

                per_layer_repr = []  # 和原来的 hs_to_save 一一对应，只是换成 text-only

                # input_ids: [B, T]
                seq_ids = inputs["input_ids"][i]
                img_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
                # print("<|vision_end|> token id",img_end_id)
                pad_id = processor.tokenizer.pad_token_id

                # 找到文本 token 的起始位置（在 </image> 后）
                if torch.is_tensor(seq_ids):
                    seq_ids_list = seq_ids.tolist()
                else:
                    seq_ids_list = [seq_ids]   # scalar → list

                # 找到 <image_end> token 的第一个位置
                try:
                    text_start = seq_ids_list.index(img_end_id) + 1
                    # print("text token 在序列中开始位置", text_start)
                    
                except ValueError:
                    text_start = 0  # 没找到 <image_end>

                for layer_hs in selected_layers:  # layer_hs: [B, T, D]
                    seq_hs = layer_hs[i]          # [T, D]

                    # 1. 取文本 token 区间
                    text_ids = seq_ids[text_start:]
                    text_hs = seq_hs[text_start:]

                    # 2. 确保设备一致
                    # 将 text_ids 移到与 text_hs 相同的设备
                    text_ids = text_ids.to(text_hs.device)
                    
                    # 创建 mask 并确保在相同设备
                    non_pad_mask = (text_ids != pad_id)
                    text_hs = text_hs[non_pad_mask]

                    if text_hs.size(0) == 0:
                        # 创建零向量时也要确保设备一致
                        per_layer_repr.append(torch.zeros(seq_hs.size(1), device=text_hs.device))
                    else:
                        per_layer_repr.append(text_hs.mean(dim=0))

                hs_to_save = per_layer_repr

                if batch_idx == 0 and i < 3:
                    print(f"[Sample {qid}-{lang}] text-mean={hs_to_save[0].mean():.4f}, std={hs_to_save[0].std():.4f}")

                cache.append((qid, lang, hs_to_save))


            # ---------------- Async save ----------------
            if len(cache) >= cache_every * batch_size:
                data_to_save = list(cache)
                cache.clear()
                pending.append(pool.submit(save_cache_batch, data_to_save, lang_save_dirs))

            t3 = time.time()
            print(f"[Batch {batch_idx}] prep={t1-t0:.2f}s | forward={t2-t1:.2f}s | cache={t3-t2:.2f}s")

        # flush remaining cache
        if cache:
            pending.append(pool.submit(save_cache_batch, list(cache), lang_save_dirs))
            cache.clear()

    for f in as_completed(pending):
        _ = f.result()

    pool.shutdown(wait=True)
    print(f"🎉 All done in {time.time() - start_time:.2f}s")

def save_cache_batch(data_list, lang_save_dirs):
    for qid, lang, hs in data_list:
        try:
            save_path = lang_save_dirs[lang] / f"sample_{qid}_{lang}.pt"
            torch.save(hs, save_path)
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