import torch
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def extract_features(model, num_samples=5000, target_layer="last", token_scope="text_last"):
    """
    Extract hidden states from LLaVA model.
    Args:
        model: LLaVA model object
        num_samples: number of examples to sample
        target_layer: which layer to extract ('last', 'penultimate', 'all')
        token_scope: 'text_last', 'text_all', 'vision_all', 'all'
    Return:
        numpy array [num_samples, dim]
    """
    # 用示例文本/图像占位（实际项目中替换为真实dataloader）
    dummy_input = "Describe the image."
    dummy_image = torch.zeros(1, 3, 336, 336).cuda()  # placeholder

    features = []
    for _ in tqdm(range(num_samples), desc="Extracting features"):
        outputs = model(dummy_image, dummy_input, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # list of [batch, seq_len, dim]
        if target_layer == "last":
            h = hidden_states[-1]
        elif target_layer == "penultimate":
            h = hidden_states[-2]
        else:
            h = torch.cat(hidden_states, dim=1).mean(1)  # 全层平均示例

        # Token 选择
        if token_scope == "text_last":
            h = h[:, -1, :]  # 最后一个文本 token
        elif token_scope == "text_all":
            h = h[:, :, :].mean(1)
        elif token_scope == "vision_all":
            h = h[:, :32, :].mean(1)  # 假设前32为视觉token
        else:  # all
            h = h.mean(1)

        features.append(h.squeeze(0).cpu())

    features = torch.stack(features)
    return features
