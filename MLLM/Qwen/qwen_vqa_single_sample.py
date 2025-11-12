#!/usr/bin/env python
# =========================================================
# 🚀 Qwen-VL-2.5 单样本多语言 hidden state 提取脚本
# =========================================================

import argparse
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)
    langs = args.langs.split(",")
    max_samples = args.max_samples
    layers_to_save = args.layers
    save_dir.mkdir(parents=True, exist_ok=True)

    print("========== CONFIG ==========")
    print(f"📁 Data path: {data_path}")
    print(f"🧠 Model path: {model_path}")
    print(f"💾 Save dir: {save_dir}")
    print(f"🌐 Langs: {langs}")
    print(f"🔢 Max samples: {max_samples}")
    print(f"📚 Layers: {layers_to_save}")
    print("============================")

    # === 加载模型 ===
    print(f"🚀 Loading model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # === 推理 ===
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_samples and idx > max_samples:
                break

            try:
                sample = json.loads(line)
                qid = sample["id"]
                img_b64 = sample["images"][0]
                image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")

                for lang in langs:
                    question = sample["questions"].get(lang)
                    if not question:
                        print(f"[⚠️] Sample {qid} missing {lang}, skip.")
                        continue

                    options = sample.get("options", [])
                    text = f"{question}\n" + "\n".join(
                        [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
                    )

                    # === 构造输入 ===
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    }]
                    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text_in],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)

                    # === 前向推理 ===
                    with torch.inference_mode():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states

                    # === 保存 hidden states ===
                    if layers_to_save == "last":
                        hs_to_save = hidden_states[-1].to(torch.float16).cpu()
                    else:
                        hs_to_save = [h.to(torch.float16).cpu() for h in hidden_states]

                    save_path = save_dir / f"sample_{qid}_{lang}.pt"
                    torch.save(hs_to_save, save_path)
                    print(f"[✅] Saved hidden state: {save_path.name}")

            except Exception as e:
                print(f"[❌] Sample {idx} failed: {e}")

    print("🎉 All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states for multilingual MDUR samples using Qwen-VL-2.5.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, default="EN,ZH,AR", help="Comma-separated list of languages.")
    parser.add_argument("--max_samples", type=int, default=1, help="Number of samples to process.")
    parser.add_argument("--layers", type=str, default="all", choices=["last", "all"], help="Which layers to save.")
    args = parser.parse_args()
    main(args)
