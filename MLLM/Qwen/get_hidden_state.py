#!/usr/bin/env python

import os
import torch
import time
import argparse
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Log function
log_file = open("log(Qwen_7B_1113).log", "a")

def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()

# Dataset Class
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

# Custom Collate Function
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

# Function to load the model and processor
def load_model(model_path, device="cuda"):
    """
    Load the model and processor for Qwen-VL based on the provided model path.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2 for acceleration
        device_map=device,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path)

    model.eval()  # Set model to evaluation mode
    processor.tokenizer.padding_side = "left"  # Set padding side to 'left'
    return model, processor

# Function to prepare input batch for inference
def prepare_input_batch(batch, processor):
    """
    Prepare a batch of input data for the model.
    """
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
    
    return inputs

# Function to perform inference
def run_inference(model, inputs):
    """
    Perform inference with the model and return the generated outputs.
    """
    try:
        outputs = model(**inputs, output_hidden_states=True)
        return outputs
    except Exception as e:
        log(f"[ERROR] Inference failed: {repr(e)}")
        raise e

# Function to process the model output and save the results
def process_output_and_save(outputs, inputs, batch, cache, lang_save_dirs, batch_idx):
    """
    Process the outputs of the model, compute hidden states, and save them.
    """
    hidden_states = outputs.hidden_states
    selected_layers = (
        [hidden_states[-1]]
        if layer_interval == 0
        else hidden_states[::layer_interval]
    )
    selected_layers = [h.to(torch.float16).cpu() for h in selected_layers]

    for i, (qid, lang) in enumerate(zip(batch["qids"], batch["langs"])):
        per_layer_repr = []

        seq_ids = inputs["input_ids"][i]
        img_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")  # Get the vision end token id
        pad_id = processor.tokenizer.pad_token_id

        # Find the starting position of the text tokens
        try:
            text_start = seq_ids.tolist().index(img_end_id) + 1
        except ValueError:
            text_start = 0  # In case no <|vision_end|> token is found

        for layer_hs in selected_layers:
            seq_hs = layer_hs[i]
            text_ids = seq_ids[text_start:]
            text_hs = seq_hs[text_start:]

            text_ids = text_ids.to(text_hs.device)
            non_pad_mask = (text_ids != pad_id)
            text_hs = text_hs[non_pad_mask]

            if text_hs.size(0) == 0:
                per_layer_repr.append(torch.zeros(seq_hs.size(1), device=text_hs.device))
            else:
                per_layer_repr.append(text_hs.mean(dim=0))

        hs_to_save = per_layer_repr

        if batch_idx == 0 and i < 3:
            print(f"[Sample {qid}-{lang}] text-mean={hs_to_save[0].mean():.4f}, std={hs_to_save[0].std():.4f}")

        cache.append((qid, lang, hs_to_save))

# Function to save cached results asynchronously
def save_cache_batch(data_list, lang_save_dirs):
    for qid, lang, hs in data_list:
        try:
            save_path = lang_save_dirs[lang] / f"sample_{qid}_{lang}.pt"
            torch.save(hs, save_path)
        except Exception as e:
            print(f"[⚠️] Failed to save {qid}_{lang}: {e}")

# Main function to orchestrate the data loading, model inference, and result saving
def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    langs = args.langs.split(",")
    max_samples = args.max_samples
    layer_interval = args.layer_interval
    cache_every = 8

    batch_size = len(langs)  # Batch size is the number of languages

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CONFIG ==========")
    print(f"langs={langs}, batch={batch_size}, layer_interval={layer_interval}, cache_every={cache_every}\n")

    print(f"Loading model from {model_path}")
    model, processor = load_model(model_path)

    # Load dataset
    dataset = PM4BenchVQA(data_path, langs, max_samples)

    # Group samples by qid
    grouped = defaultdict(list)
    for idx, s in enumerate(dataset.samples):
        grouped[s["qid"]].append(idx)
    batches = list(grouped.values())

    print(f"\nTotal qids = {len(batches)}, Will run {len(batches)} batches.")

    # Prepare for async save
    lang_save_dirs = {}
    for lang in langs:
        d = save_dir / lang
        d.mkdir(parents=True, exist_ok=True)
        lang_save_dirs[lang] = d

    pool = ThreadPoolExecutor(max_workers=4)
    pending = []
    cache = []
    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Multilingual Batches")):
            if batch is None:
                continue

            # Prepare inputs for the batch
            inputs = prepare_input_batch(batch, processor)

            # Run inference
            outputs = run_inference(model, inputs)

            # Process outputs and save results
            process_output_and_save(outputs, inputs, batch, cache, lang_save_dirs, batch_idx)

            if len(cache) >= cache_every * batch_size:
                data_to_save = list(cache)
                cache.clear()
                pending.append(pool.submit(save_cache_batch, data_to_save, lang_save_dirs))

            # Print performance stats
            print(f"[Batch {batch_idx}] Inference done.")

        # Flush remaining cache
        if cache:
            pending.append(pool.submit(save_cache_batch, list(cache), lang_save_dirs))
            cache.clear()

    for f in as_completed(pending):
        _ = f.result()

    pool.shutdown(wait=True)
    print(f"🎉 All done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states for multilingual MDUR samples using Qwen-VL-2.5.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, default="EN,ZH,AR", help="Comma-separated list of languages.")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to process. Default: all.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--layer_interval", type=int, default=1, help="Save every N layers (e.g., 2→save layers 0,2,4,...; 0→only last layer).")
    args = parser.parse_args()
    main(args)
