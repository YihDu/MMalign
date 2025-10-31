# Metrics used in Representation space

import numpy as np
from scipy.stats import linregress

def compute_rankme(eigvals, eps=1e-12):
    eigvals = np.clip(eigvals, eps, None)
    p = eigvals / eigvals.sum()
    entropy = -(p * np.log(p)).sum()
    return np.exp(entropy)

def compute_alpha_req(eigvals, fit_range=None):
    eigvals = np.array(eigvals)
    eigvals = eigvals[eigvals > 1e-12]
    log_i = np.log(np.arange(1, len(eigvals) + 1))
    log_l = np.log(eigvals)
    if fit_range is not None:
        log_i, log_l = log_i[fit_range], log_l[fit_range]
    slope, intercept, *_ = linregress(log_i, log_l)
    return -slope
