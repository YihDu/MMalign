# -*- coding: utf-8 -*-
# 表征空间的谱指标计算函数

import numpy as np
from scipy.stats import linregress


def _normalize_spectrum(spectrum, eps=1e-12):
    """
    将谱值裁剪到 eps 以上并归一化。
    TODO: 根据样本量自适应调整 eps，避免人为偏移。
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.clip(spectrum, eps, None)
    total = spectrum.sum()
    if total <= 0:
        return spectrum * 0.0
    return spectrum / total


def compute_rankme(singular_vals, eps=1e-12):
    """
    RankMe 使用奇异值分布的熵指数衡量有效秩。
    """
    probs = _normalize_spectrum(singular_vals, eps=eps)
    entropy = -(probs * np.log(probs + 1e-32)).sum()
    return float(np.exp(entropy))


def compute_effective_rank(singular_vals, eps=1e-12):
    """
    eRank 与 RankMe 定义相同，这里保留独立接口便于扩展。
    TODO: 增加其它有效秩定义（如 log-rank 或 participation ratio）。
    """
    return compute_rankme(singular_vals, eps=eps)


def compute_alpha_req(eigvals, fit_range=None, eps=1e-12):
    """
    在 log-log 坐标下拟合谱尾部斜率，衡量谱的幂律指数。
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[eigvals > eps]
    if eigvals.size < 2:
        # 样本不足时返回 NaN，交由上层决定是否跳过
        return float("nan")

    log_i = np.log(np.arange(1, eigvals.size + 1))
    log_l = np.log(eigvals)
    if fit_range is not None:
        log_i, log_l = log_i[fit_range], log_l[fit_range]

    if log_i.size < 2:
        # TODO: 提供更健壮的区间选择策略
        return float("nan")

    slope, _, _, _, _ = linregress(log_i, log_l)
    return float(-slope)
