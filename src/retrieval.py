"""Exact chance, tied-rank retrieval and candidate-inspection evaluation."""
from math import comb
import numpy as np

def chance(pool, targets, k):
    k = min(k, pool)
    return 1 - (comb(pool-targets, k) / comb(pool, k)
                if k <= pool-targets else 0.)

def query_stats(scores, targets, budget=5):
    """Expected first-target rank and hits under uniform score-tie ordering."""
    pool = len(scores)
    targets = np.asarray(targets, int)
    best = scores[targets].max()
    higher = int((scores > best).sum())
    tied = int((scores == best).sum())
    true_tied = int((scores[targets] == best).sum())
    rank = higher + (tied + 1) / (true_tied + 1)
    def hit(k):
        return chance(tied, true_tied, max(0, min(tied, k-higher)))
    b = min(budget, pool)
    capped = sum(1-hit(k) for k in range(b))
    fallback = capped + (1-hit(b)) * (pool-b+1)/(len(targets)+1)
    return dict(rank=rank, hit1=hit(1), hit5=hit(b),
                random_rank=(pool+1)/(len(targets)+1),
                random_hit5=chance(pool, len(targets), b),
                capped_inspections=capped, fallback_inspections=fallback,
                pool=pool, targets=len(targets))

