import torch
from transformers import AutoModelForCausalLM

# Todo
def load_llava_ckpt(ckpt_path):
    print(f"[INFO] Loading model from {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    return model
