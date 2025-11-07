# /experiment/experiment_design.py
import torch
from scipy.spatial.distance import cosine
from utils.helper_functions import compute_embeddings


def run_experiment(model, processor, images, captions_dict):
    """
    执行实验：为每个输入（图像和多语言的文本）计算三个对照组的对齐效果。
    
    :param model: 已加载的多模态模型
    :param processor: 模型的处理器
    :param images: 图像数据
    :param captions_dict: 每个图像对应的多语言描述字典，格式为：{language_code: caption}
    
    :return: 每个实验组的余弦距离
    """
    results = []

    # 遍历每个图像和对应的多语言描述
    for image, captions in zip(images, captions_dict):
        # 运行三组对照实验
        baseline_dist = baseline_condition(model, processor, captions, image)
        anchored_dist = correct_anchoring_condition(model, processor, captions, image)
        mismatched_dist = mismatched_anchoring_condition(model, processor, captions, image)

        results.append((baseline_dist, anchored_dist, mismatched_dist))

    return results


def baseline_condition(model, processor, captions, image):
    """
    无锚定基线实验：只使用文本输入计算多语言描述的对齐度（余弦距离）。
    
    :param model: 已加载的多模态模型
    :param processor: 模型的处理器
    :param captions: 多语言描述字典，格式为 {language_code: caption}
    :param image: 输入的图像
    
    :return: 描述之间的余弦距离
    """
    embeddings = {}
    # 获取所有语言的文本嵌入
    for language, caption in captions.items():
        embedding = compute_embeddings(model, processor, caption, language=language)  # 不同语言的描述
        embeddings[language] = embedding

    # 计算所有语言描述之间的对齐度（余弦距离）
    languages = list(embeddings.keys())
    baseline_distances = []
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            baseline_distances.append(cosine(embeddings[languages[i]], embeddings[languages[j]]))

    # 返回计算出的余弦距离
    return baseline_distances


def correct_anchoring_condition(model, processor, captions, image):
    """
    正确锚定实验：图像和文本输入一同计算描述之间的对齐度（余弦距离）。
    
    :param model: 已加载的多模态模型
    :param processor: 模型的处理器
    :param captions: 多语言描述字典，格式为 {language_code: caption}
    :param image: 输入的图像
    
    :return: 图像与文本描述的余弦距离
    """
    embeddings = {}
    # 获取所有语言的文本嵌入，结合图像进行处理
    for language, caption in captions.items():
        embedding = compute_embeddings(model, processor, caption, image, language=language)
        embeddings[language] = embedding

    # 计算所有语言描述与图像之间的对齐度（余弦距离）
    languages = list(embeddings.keys())
    anchored_distances = []
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            anchored_distances.append(cosine(embeddings[languages[i]], embeddings[languages[j]]))

    # 返回计算出的余弦距离
    return anchored_distances


def mismatched_anchoring_condition(model, processor, captions, image):
    """
    错误锚定实验：使用与图像无关的文本描述计算对齐度（余弦距离）。
    
    :param model: 已加载的多模态模型
    :param processor: 模型的处理器
    :param captions: 多语言描述字典，格式为 {language_code: caption}
    :param image: 输入的图像
    
    :return: 图像和描述之间的对齐度
    """
    embeddings = {}
    # 获取所有语言的文本嵌入，结合无关图像进行处理
    for language, caption in captions.items():
        embedding = compute_embeddings(model, processor, caption, image, language=language)
        embeddings[language] = embedding

    # 计算所有语言描述与图像之间的对齐度（余弦距离）
    languages = list(embeddings.keys())
    mismatched_distances = []
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            mismatched_distances.append(cosine(embeddings[languages[i]], embeddings[languages[j]]))

    # 返回计算出的余弦距离
    return mismatched_distances
