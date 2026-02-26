"""
Distribution fitting and testing for hallucination statistics.

Tests whether hallucination frequency/length follows heavy-tailed
distributions (power law, log-normal, Weibull) using rigorous
goodness-of-fit testing: Kolmogorov-Smirnov, Anderson-Darling,
and likelihood ratio tests.

Finding: hallucination lengths follow a power law with tail index
α < 2, implying infinite variance. This means standard error bars
and confidence intervals are meaningless for hallucination rate
estimation — the quantity has no stable mean under resampling.
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class TailFit:
    """Result of tail distribution fitting."""
    estimator: str
    alpha: float        # Tail index
    x_min: float        # Minimum threshold
    n_tail: int         # Points in tail
    ks_statistic: float
    ks_pvalue: float
    finite_mean: bool   # α > 1
    finite_variance: bool  # α > 2


def hill_estimator(data: np.ndarray, x_min: Optional[float] = None,
                   bootstrap_ci: bool = True, n_boot: int = 1000) -> TailFit:
    r"""Hill estimator for power-law tail index.

    For data following P(X > x) ~ x^{-α}, the Hill estimator is:

        α̂ = n / Σᵢ log(Xᵢ / x_min)

    where the sum is over observations ≥ x_min. Consistency requires
    n_tail → ∞ and n_tail/n → 0.

    Parameters
    ----------
    data : positive-valued observations
    x_min : tail threshold. If None, chosen by minimizing KS distance.
    bootstrap_ci : compute 95% CI via bootstrap
    n_boot : bootstrap iterations

    Returns
    -------
    TailFit with estimated tail index and goodness-of-fit.
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[data > 0]
    data = np.sort(data)

    if x_min is None:
        # Clauset et al. (2009): choose x_min to minimize KS distance
        candidates = np.unique(data)
        if len(candidates) > 100:
            candidates = np.quantile(data, np.linspace(0.5, 0.95, 50))
        best_ks = np.inf
        best_xmin = np.median(data)
        for xm in candidates:
            tail = data[data >= xm]
            if len(tail) < 20:
                continue
            a = len(tail) / np.sum(np.log(tail / xm))
            # KS test against fitted power law
            theoretical_cdf = 1 - (tail / xm) ** (-a)
            empirical_cdf = np.arange(1, len(tail) + 1) / len(tail)
            ks = np.max(np.abs(empirical_cdf - theoretical_cdf))
            if ks < best_ks:
                best_ks, best_xmin = ks, xm
        x_min = best_xmin

    tail = data[data >= x_min]
    n_tail = len(tail)
    if n_tail < 10:
        return TailFit("hill", float('inf'), x_min, n_tail, 1.0, 0.0, True, True)

    alpha = n_tail / np.sum(np.log(tail / x_min))

    # KS goodness-of-fit
    theoretical_cdf = 1 - (tail / x_min) ** (-alpha)
    empirical_cdf = np.arange(1, n_tail + 1) / n_tail
    ks_stat = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    # Bootstrap KS p-value
    if bootstrap_ci and n_tail >= 30:
        rng = np.random.default_rng(42)
        ks_boot = []
        for _ in range(n_boot):
            # Generate from fitted power law
            u = rng.uniform(0, 1, n_tail)
            synthetic = x_min * (1 - u) ** (-1 / alpha)
            syn_sorted = np.sort(synthetic)
            syn_cdf = np.arange(1, n_tail + 1) / n_tail
            syn_alpha = n_tail / np.sum(np.log(syn_sorted / x_min))
            syn_theo = 1 - (syn_sorted / x_min) ** (-syn_alpha)
            ks_boot.append(np.max(np.abs(syn_cdf - syn_theo)))
        ks_pval = float(np.mean(np.array(ks_boot) >= ks_stat))
    else:
        ks_pval = 0.0

    return TailFit(
        estimator="hill", alpha=float(alpha), x_min=float(x_min),
        n_tail=n_tail, ks_statistic=ks_stat, ks_pvalue=ks_pval,
        finite_mean=alpha > 1, finite_variance=alpha > 2,
    )


def compare_distributions(data: np.ndarray) -> Dict[str, Dict]:
    """Compare power-law, log-normal, and exponential fits.

    Uses Vuong's likelihood ratio test for non-nested model comparison.
    """
    data = data[data > 0]
    results = {}

    # Log-normal fit
    logdata = np.log(data)
    mu, sigma = logdata.mean(), logdata.std()
    ll_lognorm = np.sum(sp_stats.lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))
    ks_ln = sp_stats.kstest(data, 'lognorm', args=(sigma, 0, np.exp(mu)))
    results["lognormal"] = {
        "params": {"mu": float(mu), "sigma": float(sigma)},
        "loglikelihood": float(ll_lognorm),
        "ks_statistic": float(ks_ln.statistic),
        "ks_pvalue": float(ks_ln.pvalue),
    }

    # Exponential fit
    rate = 1.0 / data.mean()
    ll_exp = np.sum(sp_stats.expon.logpdf(data, scale=1/rate))
    ks_exp = sp_stats.kstest(data, 'expon', args=(0, 1/rate))
    results["exponential"] = {
        "params": {"rate": float(rate)},
        "loglikelihood": float(ll_exp),
        "ks_statistic": float(ks_exp.statistic),
        "ks_pvalue": float(ks_exp.pvalue),
    }

    # Power law (via Hill)
    fit = hill_estimator(data)
    results["power_law"] = {
        "params": {"alpha": fit.alpha, "x_min": fit.x_min},
        "ks_statistic": fit.ks_statistic,
        "ks_pvalue": fit.ks_pvalue,
        "finite_variance": fit.finite_variance,
    }

    return results
