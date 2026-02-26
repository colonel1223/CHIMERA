"""
Entropy and mutual information estimation for trace analysis.

Measures the information content of model outputs and the mutual
information between confidence scores and correctness. Low MI
indicates the model's confidence is uninformative about its accuracy
— the signature of poor calibration.
"""

import numpy as np
from typing import Optional


def shannon_entropy(probs: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy H(X) = -Σ p(x) log p(x)."""
    p = np.asarray(probs, dtype=np.float64).flatten()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) = Σ p(x) log(p(x)/q(x)). Returns nats."""
    p = np.asarray(p, dtype=np.float64).flatten()
    q = np.asarray(q, dtype=np.float64).flatten()
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def mutual_information_binned(x: np.ndarray, y: np.ndarray,
                               n_bins: int = 20) -> float:
    """Estimate MI(X; Y) via binning.

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    For continuous variables, discretizes into n_bins equal-frequency bins.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Equal-frequency binning
    x_bins = np.searchsorted(np.sort(x), x) * n_bins // len(x)
    y_bins = np.searchsorted(np.sort(y), y) * n_bins // len(y)

    # Joint histogram
    joint = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x_bins, y_bins):
        xi = min(xi, n_bins - 1)
        yi = min(yi, n_bins - 1)
        joint[xi, yi] += 1
    joint /= joint.sum()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = H(X) + H(Y) - H(X,Y)
    hx = shannon_entropy(px)
    hy = shannon_entropy(py)
    hxy = shannon_entropy(joint)

    return float(max(hx + hy - hxy, 0.0))


def confidence_informativeness(confidences: np.ndarray,
                                correct: np.ndarray) -> dict:
    """Measure how informative confidence scores are about correctness.

    Returns MI(confidence; correct) and normalized MI (0 = useless, 1 = perfect).
    """
    mi = mutual_information_binned(confidences, correct.astype(float))
    h_correct = shannon_entropy(np.array([correct.mean(), 1 - correct.mean()]))

    return {
        "mutual_information_bits": mi,
        "normalized_mi": mi / h_correct if h_correct > 0 else 0.0,
        "h_correct": h_correct,
        "interpretation": (
            "strong signal" if mi / max(h_correct, 1e-10) > 0.3
            else "weak signal" if mi / max(h_correct, 1e-10) > 0.1
            else "near-useless"
        ),
    }
