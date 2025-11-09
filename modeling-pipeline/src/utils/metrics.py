# src/utils/metrics.py
import numpy as np
from scipy.stats import pearsonr, spearmanr

def _to1d(x):
    a = np.asarray(x)
    return a.reshape(-1)

def _safe_div(a, b, eps=1e-8):
    return a / np.maximum(np.abs(b), eps)

def rmse(y, p):
    y, p = _to1d(y), _to1d(p)
    return float(np.sqrt(np.mean((p - y) ** 2)))

def mae(y, p):
    y, p = _to1d(y), _to1d(p)
    return float(np.mean(np.abs(p - y)))

def r2(y, p):
    y, p = _to1d(y), _to1d(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / np.maximum(ss_tot, 1e-12))

def mape(y, p, eps=1e-6):
    y, p = _to1d(y), _to1d(p)
    return float(np.mean(np.abs(_safe_div(p - y, y, eps))) * 100.0)

def smape(y, p, eps=1e-6):
    y, p = _to1d(y), _to1d(p)
    return float(np.mean(2.0 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), eps)) * 100.0)

def bias(y, p):
    y, p = _to1d(y), _to1d(p)
    return float(np.mean(p - y))

def nrmse_mean(y, p):
    y, p = _to1d(y), _to1d(p)
    return float(rmse(y, p) / (np.mean(y) + 1e-8))

def nrmse_iqr(y, p):
    y, p = _to1d(y), _to1d(p)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = max(q3 - q1, 1e-8)
    return float(rmse(y, p) / iqr)

def acc_within(y, p, pct=10.0, eps=1e-6):
    y, p = _to1d(y), _to1d(p)
    ok = np.abs(_safe_div(p - y, y, eps)) * 100.0 <= pct
    return float(np.mean(ok) * 100.0)

def pearson_r(y, p):
    y, p = _to1d(y), _to1d(p)
    if y.size < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return 0.0
    return float(pearsonr(y, p)[0])

def spearman_rho(y, p):
    y, p = _to1d(y), _to1d(p)
    if y.size < 2:
        return 0.0
    return float(spearmanr(y, p).correlation)

def explained_variance(y, p):
    y, p = _to1d(y), _to1d(p)
    var_y = np.var(y)
    return float(1.0 - np.var(y - p) / np.maximum(var_y, 1e-12))

def metrics_extended(y, p):
    y, p = _to1d(y), _to1d(p)
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "mape%": mape(y, p),
        "smape%": smape(y, p),
        "bias": bias(y, p),
        "nrmse_mean": nrmse_mean(y, p),
        "nrmse_iqr": nrmse_iqr(y, p),
        "pearson_r": pearson_r(y, p),
        "spearman_rho": spearman_rho(y, p),
        "explained_variance": explained_variance(y, p),
        "acc@10%": acc_within(y, p, 10.0),
        "acc@20%": acc_within(y, p, 20.0),
        "acc@30%": acc_within(y, p, 30.0),
        "n": int(y.size),
    }
