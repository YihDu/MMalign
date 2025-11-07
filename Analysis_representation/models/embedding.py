"""Lightweight helpers to extract normalised embeddings from multimodal models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
from PIL import Image
import torch
import numpy as np

from data.schemas import MultilingualExample


EmbeddingMap = Dict[str, np.ndarray]


def encode_examples(
    examples: Iterable[MultilingualExample],
    model,
    processor,
) -> Tuple[EmbeddingMap, np.ndarray]:
    """Return language->embedding matrix and image embeddings for the provided examples."""

    texts_by_language: dict[str, list[str]] = {}
    image_inputs: List[object] = []
    example_list = list(examples)
    for example in example_list:
        pil_image = example.image.to_model_input()
        # 将图像转换为 Processor 可直接消费的输入（PIL 或路径）
        image_inputs.append(np.array(pil_image.convert("RGB"), dtype=np.uint8))
        for language, caption in example.captions.items():
            # 逐语言累积文本，保持与图像顺序一致
            texts_by_language.setdefault(language, []).append(caption.text)
    
    # for idx, img in enumerate(image_inputs):
    #     print(idx, type(img), getattr(img, "mode", None), getattr(img, "size", None))
    
    # ==================== 调试代码：开始 ====================
    print("--- 正在检查图像数据一致性 ---")
    is_consistent = True
    first_shape = None
    for idx, img_array in enumerate(image_inputs):
        if not isinstance(img_array, np.ndarray):
            print(f"错误: 索引 {idx} 的元素不是一个 NumPy 数组，而是 {type(img_array)}")
            is_consistent = False
            continue

        if first_shape is None:
            first_shape = img_array.shape
        
        print(f"图像索引 {idx}: 类型={img_array.dtype}, 形状={img_array.shape}")

        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            print(f"==> 警告: 图像索引 {idx} 的形状不是 (H, W, 3)，可能不是标准的RGB图像。")
            is_consistent = False

    if is_consistent:
        print("--- 图像数据形状和通道检查通过 ---")
    else:
        print("!!! 错误: 检测到图像数据不一致，请检查以上警告/错误信息。这将导致后续处理失败。 !!!")
    # ==================== 调试代码：结束 ====================

    image_embeddings = _encode_images(images=image_inputs, model=model, processor=processor)
    text_embeddings = {
        language: _encode_texts(texts, model, processor)
        for language, texts in texts_by_language.items()
    }
    return text_embeddings, image_embeddings


def _encode_images(images, model, processor, normalize=True, debug=True):
    """
    最终稳定调试版: 修复 CLIPImageProcessor batch bug + 兼容不同返回类型
    """
    from PIL import Image
    import numpy as np
    import torch

    if debug:
        print("\n==================== DEBUG _encode_images (final v2) ====================")

    # --- Step 1: 标准化输入 ---
    if isinstance(images, (np.ndarray, Image.Image)):
        images = [images]

    target_size = processor.image_processor.crop_size
    target_w = target_size.get("width", list(target_size.values())[0])
    target_h = target_size.get("height", list(target_size.values())[0])

    pil_images = []
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            if isinstance(img, np.ndarray):
                if img.dtype == np.float32 or img.max() <= 1.0:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.shape[-1] == 4:
                    img = img[..., :3]
                img = Image.fromarray(img).convert("RGB")
        img = img.resize((target_w, target_h))
        pil_images.append(img)
        if debug:
            print(f"[{i}] -> size={img.size}, mode={img.mode}")

    # --- Step 2: processor 的预处理逻辑 ---
    raw = processor.image_processor(
        pil_images,
        do_resize=False,
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        return_tensors=None,
    )

    arrays = raw["pixel_values"]

    # --- Step 3: 手动堆叠 ---
    tensor_list = []
    for i, a in enumerate(arrays):
        # 强制转换为 numpy array
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        elif not isinstance(a, np.ndarray):
            try:
                a = np.array(a)
            except Exception:
                raise TypeError(f"Unexpected element type at index {i}: {type(a)}")
        a = np.ascontiguousarray(np.array(a, dtype=np.float32, copy=False))
        if not isinstance(a, np.ndarray) or a.dtype != np.float32:
            a = np.array(a, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(a)
        # ----------------
        tensor_list.append(tensor)

        if debug:
            print(f"[{i}] array -> type={type(a)}, shape={a.shape}, dtype={a.dtype}")

    pixel_values = torch.stack(tensor_list, dim=0).to(model.device, dtype=model.dtype)

    if debug:
        print(f"✅ pixel_values.stack OK -> shape={pixel_values.shape}, dtype={pixel_values.dtype}")

    # --- Step 4: 前向传播 ---
    with torch.no_grad():
        outputs = model.vision_tower(pixel_values, output_hidden_states=False)
        emb = outputs.pooler_output.detach().cpu().numpy()

    if debug:
        print(f"[INFO] embedding shape: {emb.shape}")
        print("==================== END DEBUG ====================\n")

    if normalize:
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb







# 确保您保留了那个健壮的 _normalize 函数
def _normalize(embeddings: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(embeddings)):
        print("!!! _normalize 警告: 输入的嵌入向量中包含 NaN 或 Inf，正在进行清理...")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e5, neginf=-1e5)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    clipped_norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / clipped_norms

def _encode_texts(
    texts: Sequence[str],
    model,  # LlavaForConditionalGeneration
    processor,  # LlavaProcessor
) -> np.ndarray:
    """
    接收一批文本，使用 LLaVA 模型的语言模型部分对其进行编码。
    """
    # 1. 使用 processor 对文本进行分词和张量化
    inputs = processor(text=list(texts), return_tensors="pt", padding=True)

    # 2. 将输入张量移动到模型所在的设备
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # 3. !! 核心修复：执行完整的模型前向传播，并请求输出隐藏状态 !!
    with torch.no_grad():
        # model(**inputs) 会调用语言模型
        # 对于纯文本输入，它会忽略 vision_tower
        outputs = model(**inputs, output_hidden_states=True)

    # 4. 从输出中提取最后一层的隐藏状态
    # outputs.hidden_states 是一个元组，包含了从输入层到最后一层的所有隐藏状态
    last_hidden_state = outputs.hidden_states[-1]

    # 5. 使用 "last token pooling" 策略提取句子嵌入
    # 我们需要找到每个句子中最后一个非填充 token 的位置
    # attention_mask 中值为 1 的地方是有效 token，它的和即为句子长度
    last_token_indices = inputs["attention_mask"].sum(dim=1) - 1

    # 使用这些索引从 last_hidden_state 中提取对应的向量
    batch_size = last_hidden_state.shape[0]
    text_embeddings = last_hidden_state[
        torch.arange(batch_size, device=model.device), last_token_indices
    ]
    
    # 6. 将张量转为 NumPy 数组，并进行归一化（与原代码逻辑保持一致）
    embeddings = text_embeddings.detach().cpu().numpy()  
    
    return _normalize(embeddings)

