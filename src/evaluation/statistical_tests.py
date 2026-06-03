"""Small statistical helpers for multi-seed comparisons."""

from __future__ import annotations

import math

import numpy as np


def paired_mean_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
        "n": int(diff.size),
    }


def paired_metric_statistics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    higher_is_better: bool = False,
) -> dict[str, float]:
    """Paired improvement statistics for lower-is-better metrics by default."""
    b = np.asarray(baseline, dtype=float).reshape(-1)
    c = np.asarray(candidate, dtype=float).reshape(-1)
    mask = np.isfinite(b) & np.isfinite(c)
    b = b[mask]
    c = c[mask]
    if b.size == 0:
        return _empty_stats()
    diff = (c - b) if higher_is_better else (b - c)
    pct = np.divide(diff, np.abs(b), out=np.full_like(diff, np.nan), where=np.abs(b) > 1e-12) * 100.0
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    se = std_diff / float(np.sqrt(diff.size)) if diff.size > 1 else 0.0
    ci_half = float(_t_critical_approx(diff.size - 1) * se) if diff.size > 1 else 0.0
    t_stat, t_p = _paired_t_test(diff)
    w_stat, w_p = _wilcoxon_signed_rank(diff)
    return {
        "valid_n": int(diff.size),
        "paired_improvement_percent_mean": float(np.nanmean(pct)) if np.any(np.isfinite(pct)) else float("nan"),
        "paired_difference_mean": mean_diff,
        "paired_difference_std": std_diff,
        "paired_difference_ci95_low": mean_diff - ci_half,
        "paired_difference_ci95_high": mean_diff + ci_half,
        "paired_t_stat": t_stat,
        "paired_t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
    }


def _empty_stats() -> dict[str, float]:
    return {
        "valid_n": 0,
        "paired_improvement_percent_mean": float("nan"),
        "paired_difference_mean": float("nan"),
        "paired_difference_std": float("nan"),
        "paired_difference_ci95_low": float("nan"),
        "paired_difference_ci95_high": float("nan"),
        "paired_t_stat": float("nan"),
        "paired_t_p_value": float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p_value": float("nan"),
    }


def _paired_t_test(diff: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return float("nan"), float("nan")
    std = float(np.std(diff, ddof=1))
    if std <= 1e-15:
        return float("inf") if float(np.mean(diff)) != 0.0 else 0.0, 0.0 if float(np.mean(diff)) != 0.0 else 1.0
    t_stat = float(np.mean(diff) / (std / np.sqrt(diff.size)))
    try:
        from scipy import stats  # type: ignore

        p = float(stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue)
    except Exception:
        p = float(2.0 * (1.0 - _normal_cdf(abs(t_stat))))
    return t_stat, p


def _wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-15]
    if diff.size == 0:
        return 0.0, 1.0
    try:
        from scipy import stats  # type: ignore

        result = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        ranks = _average_ranks(np.abs(diff))
        w_plus = float(np.sum(ranks[diff > 0]))
        w_minus = float(np.sum(ranks[diff < 0]))
        w = min(w_plus, w_minus)
        n = diff.size
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        if var <= 0.0:
            return w, float("nan")
        z = (w - mean) / np.sqrt(var)
        p = float(2.0 * _normal_cdf(z))
        return w, max(0.0, min(1.0, p))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(x) / np.sqrt(2.0))))


def _t_critical_approx(df: int) -> float:
    if df <= 0:
        return float("nan")
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 25:
        return table[max(k for k in table if k < df)]
    if df < 30:
        return table[25]
    return 1.96
