"""
Information-theoretic impossibility bounds on hallucination elimination.

Core argument: For open-ended generation with vocabulary V and output
length L, the space of possible outputs is |V|^L. Under any bounded
compute budget C, only C / (cost_per_verification * L) outputs can be
verified against ground truth. The fraction of unverifiable outputs
approaches 1 exponentially in L, giving a non-zero floor on
hallucination probability for ANY model architecture.

This is not a statement about current models being bad. It's a
statement about the combinatorial structure of language generation
under finite resources.

Strengthening: even with oracle access to a ground-truth verifier,
the verification bottleneck remains because the verifier itself
requires compute proportional to output length.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class ImpossibilityBound:
    """Result of impossibility bound computation."""
    vocab_size: int
    seq_len: int
    compute_budget_flops: float
    verification_cost_per_token: float
    log_output_space: float      # log2(|V|^L)
    log_verifiable: float        # log2(verifiable outputs)
    hallucination_floor: float   # Lower bound on P(hallucination)
    bits_per_token_unverified: float


def compute_impossibility_bound(
    vocab_size: int = 50_000,
    seq_len: int = 256,
    compute_budget: float = 1e18,
    flops_per_token_verify: float = 1e9,
) -> ImpossibilityBound:
    r"""Compute lower bound on achievable hallucination rate.

    Under compute budget C and per-token verification cost c:
    - Total output space: |V|^L
    - Verifiable outputs: ⌊C / (c · L)⌋
    - Fraction verifiable: C / (c · L · |V|^L)
    - Hallucination floor: 1 - C / (c · L · |V|^L)

    For any realistic parameters, this floor is indistinguishable from 1
    for long sequences, making zero-hallucination provably infeasible.

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    seq_len : int
        Output sequence length.
    compute_budget : float
        Total available FLOPs for verification.
    flops_per_token_verify : float
        FLOPs to verify one token against ground truth.
    """
    # Work in log space to avoid overflow
    log2_output_space = seq_len * np.log2(vocab_size)
    verifiable = compute_budget / (flops_per_token_verify * seq_len)
    log2_verifiable = np.log2(max(verifiable, 1))

    # Fraction of output space that can be verified
    log2_fraction = log2_verifiable - log2_output_space

    if log2_fraction >= 0:
        # Can verify everything (only for trivially small spaces)
        floor = 0.0
    else:
        floor = 1.0 - 2 ** log2_fraction

    # Bits per token that remain unverified
    bits_unverified = log2_output_space - log2_verifiable

    return ImpossibilityBound(
        vocab_size=vocab_size,
        seq_len=seq_len,
        compute_budget_flops=compute_budget,
        verification_cost_per_token=flops_per_token_verify,
        log_output_space=log2_output_space,
        log_verifiable=log2_verifiable,
        hallucination_floor=float(min(floor, 1.0)),
        bits_per_token_unverified=float(max(bits_unverified / seq_len, 0)),
    )


def scaling_analysis(vocab_size: int = 50_000,
                     compute_budget: float = 1e18,
                     flops_per_token: float = 1e9) -> Dict[int, float]:
    """How hallucination floor scales with sequence length."""
    results = {}
    for L in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        bound = compute_impossibility_bound(vocab_size, L, compute_budget, flops_per_token)
        results[L] = bound.hallucination_floor
    return results


if __name__ == "__main__":
    print("CHIMERA Impossibility Bounds")
    print("=" * 60)

    bound = compute_impossibility_bound()
    print(f"Vocabulary:            {bound.vocab_size:,}")
    print(f"Sequence length:       {bound.seq_len}")
    print(f"Compute budget:        {bound.compute_budget_flops:.0e} FLOPs")
    print(f"Log₂ output space:     {bound.log_output_space:,.0f} bits")
    print(f"Log₂ verifiable:       {bound.log_verifiable:,.1f} bits")
    print(f"Hallucination floor:   {bound.hallucination_floor:.10f}")
    print(f"Unverified bits/token: {bound.bits_per_token_unverified:.1f}")

    print(f"\nScaling with sequence length:")
    print(f"{'Length':>8} {'Floor':>20}")
    for L, floor in scaling_analysis().items():
        print(f"{L:>8} {floor:>20.10f}")
