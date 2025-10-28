import torch
from transformers import AutoModelForCausalLM

def load_llava_ckpt(ckpt_path):
    """
    加载 LLaVA 模型（示意版）。
    如果你本地有完整 LLaVA 结构，可改为:
        from llava.model.builder import load_pretrained_model
    """
    print(f"[INFO] Loading model from {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    return model
