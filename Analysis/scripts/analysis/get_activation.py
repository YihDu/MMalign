#!/usr/bin/env python
# ============================================================
# Script: get_activation.py
# Purpose: Extract hidden activations from LLaVA checkpoints
#          for spectral analysis or VL-SAE / SAE-Track training.
# Author: Yihang-style unified pipeline
# ============================================================

import sys
sys.path.append('/workspace/MMalign/MLLM/LLaVA')

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import json, os, random, yaml
from easydict import EasyDict as edict
import glob
from types import SimpleNamespace,MethodType


from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
import torch.nn as nn


# ========== utils/config_loader.py ==========
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)




def build_llava_from_vicuna_and_clip(cfg, projector_ckpt=None):
    print(f"🧠 加载 Vicuna: {cfg.model.text_model_name_or_path}")
    print(f"👁️  加载 CLIP 视觉塔: {cfg.model.vision_model_name_or_path}")

    # === 1️⃣ 加载 Vicuna backbone ===
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name_or_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        cfg.model.text_model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(cfg.device)

    # === 2️⃣ 加载 Vision Tower ===
    vision_tower_path = cfg.model.vision_model_name_or_path
    args = SimpleNamespace(
        vision_tower=vision_tower_path,
        device=cfg.device,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
    )
    vision_tower = CLIPVisionTower(vision_tower_path, args)
    vision_tower.load_model(device_map="cpu")

    model.vision_tower = vision_tower

    # ✅ 补充一个 get_vision_tower() 方法，供后续调用
    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)
    model.get_vision_tower = MethodType(get_vision_tower, model)

    # === 3️⃣ 注册并加载 projector ===
    if model.get_vision_tower() is None:
        raise RuntimeError("Vision tower 未加载成功")

    vision_dim = model.get_vision_tower().hidden_size
    text_dim = model.model.embed_tokens.weight.shape[1]
    print(f"🔧 注册 projector: Linear({vision_dim} → {text_dim})")
    model.mm_projector = nn.Linear(vision_dim, text_dim, bias=True).to(cfg.device, dtype=torch.float16)

    if projector_ckpt and os.path.exists(projector_ckpt):
        print(f"📦 加载 projector 权重: {projector_ckpt}")
        state_dict = torch.load(projector_ckpt, map_location="cpu")
        filtered = {k: v for k, v in state_dict.items() if "weight" in k or "bias" in k}
        missing, unexpected = model.mm_projector.load_state_dict(filtered, strict=False)
        print(f"✅ projector 加载成功 (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("⚠️ 未提供 projector_ckpt，仅初始化空 projector.")

    image_processor = vision_tower.image_processor
    return tokenizer, model, image_processor





# ========== 数据加载函数 ==========
def load_forward_data(json_path, image_root, num_samples=10000, seed=42):
    random.seed(seed)
    with open(json_path, "r") as f:
        ann = json.load(f)

    pairs = []
    if "annotations" in ann:  # 官方 COCO captions 格式
        id2file = {img["id"]: img["file_name"] for img in ann["images"]}
        samples = random.sample(ann["annotations"], min(num_samples, len(ann["annotations"])))
        for a in tqdm(samples, desc="Loading COCO pairs"):
            img_path = os.path.join(image_root, id2file[a["image_id"]])
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                pairs.append((img, a["caption"]))
            except Exception as e:
                print(f"⚠️ 跳过损坏图片 {img_path}: {e}")
    else:  # 自制 json
        samples = random.sample(ann, min(num_samples, len(ann)))
        for s in tqdm(samples, desc="Loading custom pairs"):
            img_path = os.path.join(image_root, s["image_path"])
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                pairs.append((img, s["caption"]))
            except Exception as e:
                print(f"⚠️ 跳过损坏图片 {img_path}: {e}")

    print(f"✅ 共加载 {len(pairs)} 个图文对")
    return pairs


@torch.no_grad()
def extract_features(model, tokenizer, image_processor, dataset,
                     target_layer="last", token_scope="text_last",
                     device="cuda", batch_size=1):
    model.eval().to(device)
    features_per_layer = {}  # {layer_idx: tensor}
    vision_tower = model.get_vision_tower()
    mm_projector = model.mm_projector  # 线性层映射视觉特征

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"[{target_layer}] Extracting"):
        batch = dataset[i : i + batch_size]
        images, texts = zip(*batch)

        # === 1️⃣ 视觉编码 ===
        pixel_values = image_processor.preprocess(list(images), return_tensors="pt")["pixel_values"].to(device)
        vision_feats = vision_tower(pixel_values)                     # [B, num_patches, dim_v]
        vision_feats = vision_feats.to(dtype=next(mm_projector.parameters()).dtype)  # ✅ 保证dtype一致
        vision_embeds = mm_projector(vision_feats)   

        # === 2️⃣ 文本编码 ===
        inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = model.model.embed_tokens(inputs["input_ids"])  # [B, seq_len, dim_text]

        # === 3️⃣ 拼接图文 token（模拟 LLaVA 输入）
        input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=None,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)

        # 遍历每一层
        for layer_idx, h in enumerate(hidden_states):
            if token_scope == "text_last":
                h = h[:, -1, :]
            elif token_scope == "text_all":
                h = h.mean(1)
            else:
                raise ValueError(f"Unknown token_scope {token_scope}")

            if layer_idx not in features_per_layer:
                features_per_layer[layer_idx] = [h.cpu()]
            else:
                features_per_layer[layer_idx].append(h.cpu())

    # 拼接所有batch
    for k in features_per_layer:
        features_per_layer[k] = torch.cat(features_per_layer[k], dim=0)

    return features_per_layer  # dict: {layer_idx: [N, D]}


# ========== 主入口 ==========
if __name__ == "__main__":
    import argparse
    from llava.model.builder import load_pretrained_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg.get("mode", "spectral")  # 默认旧逻辑

    

    
    # 加载数据
    dataset = load_forward_data(
        json_path=cfg.paths.forward_data_path_json,
        image_root=cfg.paths.forward_data_path,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
    )
    

    # 3️⃣ 模式分支
    if mode == "spectral":
        # === 原始谱分析逻辑：单层 + 单 token_scope ===
        feats = extract_features(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset=dataset,
            target_layer=cfg.model.target_layer,
            token_scope=cfg.model.token_scope,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )
        os.makedirs(cfg.paths.activation_save_dir, exist_ok=True)
        out_path = os.path.join(
            cfg.paths.activation_save_dir,
            f"features_{cfg.model.target_layer}_{len(dataset)}.npy",
        )
        np.save(out_path, feats.numpy())
        print(f"✅ [Spectral] Saved {feats.shape} to {out_path}")

    elif mode == "sae":
        # print("当前工作目录:", os.getcwd())
        # === SAE 模式：跨层 × 跨 ckpt × vision/text 双模态 ===
        ckpt_path_folder = cfg.model.get("ckpt_path_folder",None)
        layers = cfg.model.get("layers", [cfg.model.layers])
        token_scopes = cfg.model.get("token_scopes", ["vision_all", "text_all"])

        print(f"读取文件夹 {ckpt_path_folder} 中checkpoint")
        
        if ckpt_path_folder:
            
            pattern = os.path.join(ckpt_path_folder, "checkpoint-*", "mm_projector.bin")
            ckpt_list = sorted(glob.glob(pattern))      
            print(f"一共 {len(ckpt_list)} 个checkpoint")      

        else:
            raise ValueError("ckpt路径错误")
            
        for ckpt in tqdm(ckpt_list):
            print(f"📦 Loading checkpoint: {ckpt}")
            
            tokenizer, model, image_processor = build_llava_from_vicuna_and_clip(cfg, ckpt)

            for layer in layers:
                for token_scope in token_scopes:
                    feats_dict = extract_features(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        dataset=dataset,
                        token_scope=token_scope,
                        device=cfg.device,
                        batch_size=cfg.batch_size,
                    )

                    tag_ckpt = os.path.basename(os.path.dirname(ckpt))
                    out_dir = cfg.paths.activation_save_dir
                    os.makedirs(out_dir, exist_ok=True)

                    for layer_idx, feats in feats_dict.items():
                        out_name = f"activ_layer{layer_idx}_{token_scope}_{tag_ckpt}.npy"
                        np.save(os.path.join(out_dir, out_name), feats.numpy())
                        print(f"✅ [SAE] Saved {feats.shape} from layer {layer_idx} to {out_name}")
                
        else:
            raise ValueError("mode 错误")

