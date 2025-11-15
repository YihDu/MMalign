#!/usr/bin/env python

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
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from qwen_vl_utils import process_vision_info


# utilities
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

                pil_images = [
                    Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    for b64 in s["images"]
                ]

                main_img = pil_images[0]

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

class LlavaNextWrapper:
    """和 Qwen wrapper 一样结构，但符合 LLaVA-Next 的输入规范"""

    def __init__(self, model_path):
        print(f"Loading LLaVA-Next from {model_path}")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cuda",
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model.eval()

        print("LLaVA-Next Loaded!\n")

    def prepare_inputs(self, batch):
        """LLaVA 输入与 Qwen 输入不同，这里按 LLaVA-Next 的格式处理"""
        images = batch["images"]
        texts = batch["texts"]

        inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        return inputs

    def forward(self, inputs):
        return self.model(**inputs, output_hidden_states=True)

    def get_special_ids(self):
        """
        LLaVA 没有 <|vision_start|>/<|vision_end|> 这种 token
        
        - pad_id 依然从 tokenizer 中取
        - vision_end 我们标记为 None（主逻辑会跳过 vision token search）
        """
        pad_id = self.processor.tokenizer.pad_token_id
        return None, None, pad_id

class QwenVLModelWrapper:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()
        print("Model loaded!\n")

    def prepare_inputs(self, batch):
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
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        image_inputs, video_inputs = process_vision_info(messages_list)

        inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=8192,
        ).to("cuda")

        return inputs

    def forward(self, inputs):
        return self.model(**inputs, output_hidden_states=True)

    def get_special_ids(self):
        vstart = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vend = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        pad_id = self.processor.tokenizer.pad_token_id
        return vstart, vend, pad_id

def save_cache_batch(data_list, lang_save_dirs):
    for qid, lang, hs in data_list:
        try:
            save_path = lang_save_dirs[lang] / f"sample_{qid}_{lang}.pt"
            torch.save(hs, save_path)
        except Exception as e:
            print(f"[⚠️] Failed to save {qid}_{lang}: {e}")



def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    langs = args.langs.split(",")
    max_samples = args.max_samples
    layer_interval = args.layer_interval
    cache_every = 8
    batch_size = len(langs)

    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== CONFIG ==========")
    print(f"langs={langs}, batch={batch_size}, layer_interval={layer_interval}, cache_every={cache_every}\n")

    model_wrapper = QwenVLModelWrapper(model_path)

    dataset = PM4BenchVQA(data_path, langs, max_samples)

    from collections import defaultdict
    grouped = defaultdict(list)
    for idx, s in enumerate(dataset.samples):
        grouped[s["qid"]].append(idx)
    batches = list(grouped.values())

    print(f"\nTotal qids = {len(batches)}, Will run {len(batches)} batches.")

    class ManualBatchLoader:
        def __init__(self, dataset, batches):
            self.dataset = dataset
            self.batches = batches
        def __len__(self):
            return len(self.batches)
        def __iter__(self):
            for b in self.batches:
                yield custom_collate_fn([self.dataset[i] for i in b])

    loader = ManualBatchLoader(dataset, batches)

    # make per-lang dirs
    lang_save_dirs = {lang: (save_dir / lang) for lang in langs}
    for d in lang_save_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    pool = ThreadPoolExecutor(max_workers=4)
    pending = []
    cache = []
    start_time = time.time()

    vstart, vend, pad_id = model_wrapper.get_special_ids()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Multilingual Batches")):

            t0 = time.time()
            inputs = model_wrapper.prepare_inputs(batch)
            t1 = time.time()

            try:
                outputs = model_wrapper.forward(inputs)
            except Exception as e:
                err = str(e)
                if "Image features and image tokens do not match" in err:
                    print(f"[⚠️ Skip] Batch {batch_idx} token mismatch.")
                    torch.cuda.empty_cache()
                    continue
                raise

            hidden_states = outputs.hidden_states
            t2 = time.time()

            selected_layers = (
                [hidden_states[-1]]
                if layer_interval == 0
                else hidden_states[::layer_interval]
            )
            selected_layers = [h.to(torch.float16).cpu() for h in selected_layers]

            # 对每个样本取 mean pooling
            # 从Vision token + 1
            for i, (qid, lang) in enumerate(zip(batch["qids"], batch["langs"])):

                seq_ids = inputs["input_ids"][i]
                seq_ids_list = seq_ids.tolist()

                try:
                    text_start = seq_ids_list.index(vend) + 1
                except ValueError:
                    text_start = 0

                per_layer_repr = []
                for layer_hs in selected_layers:
                    seq_hs = layer_hs[i]

                    text_ids = seq_ids[text_start:]
                    text_hs = seq_hs[text_start:]
                    text_ids = text_ids.to(text_hs.device)

                    non_pad_mask = (text_ids != pad_id)
                    text_hs = text_hs[non_pad_mask]

                    if text_hs.size(0) == 0:
                        per_layer_repr.append(torch.zeros(seq_hs.size(1)))
                    else:
                        per_layer_repr.append(text_hs.mean(dim=0))

                hs_to_save = per_layer_repr

                if batch_idx == 0 and i < 3:
                    print(f"[Sample {qid}-{lang}] text-mean={hs_to_save[0].mean():.4f}, std={hs_to_save[0].std():.4f}")

                cache.append((qid, lang, hs_to_save))

            if len(cache) >= cache_every * batch_size:
                pending.append(
                    pool.submit(save_cache_batch, list(cache), lang_save_dirs)
                )
                cache.clear()

            t3 = time.time()
            print(f"[Batch {batch_idx}] prep={t1-t0:.2f}s | forward={t2-t1:.2f}s | cache={t3-t2:.2f}s")

        if cache:
            pending.append(pool.submit(save_cache_batch, list(cache), lang_save_dirs))
            cache.clear()

    for f in as_completed(pending):
        _ = f.result()

    pool.shutdown(wait=True)
    print(f"🎉 All done in {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract hidden states for multilingual MDUR samples using Qwen-VL-2.5."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, default="EN,ZH,AR")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layer_interval", type=int, default=1)

    args = parser.parse_args()
    main(args)
