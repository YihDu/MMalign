# -*- coding: utf-8 -*-
"""
通用的配置读取工具，便于在分析/绘图等脚本中统一加载 YAML。
"""

import os
import yaml
from easydict import EasyDict as edict


def load_config(path):
    """
    加载并校验配置文件路径。
    返回 EasyDict，使得可以通过属性访问配置项。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        # TODO: 支持空配置的默认字段填充
        raise ValueError(f"配置文件内容为空: {path}")

    return edict(cfg)

