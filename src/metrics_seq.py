"""Evaluation for the binding-order reconstruction task.

Four questions are asked of every predicted order, and they are not
interchangeable:

  Kendall tau / Spearman rho   Is the *global* ordering right, on average?
  Adjacent-pair accuracy       Are the *local* neighbour relations right --
                                the ones a reader actually depends on?
  Longest correct run (LIS)    How long a stretch can be trusted *as read*,
                                without spot-checking against the rest?

A reading order recovered with high Kendall tau but a short longest correct
run is still unusable to a philologist, who reads slips in a block, not as an
unordered set of pairwise-correct neighbours -- the same distinction
``yunmeng``'s ``metrics.py`` draws between join F1 and whole-slip recovery.

Reading direction is never observable from physical or textual evidence alone
-- nothing in notch drift, length trim or topical coherence says which end of
a roll was read first. Every metric here is therefore evaluated in both
orientations and the better one is reported per manuscript, which is the
standard convention for seriation and is what a human transcriber does
implicitly by reading the sensible way once the order is otherwise fixed.
"""
from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import adjusted_rand_score


def _lis_length(seq: list[int]) -> int:
    """Length of the longest strictly increasing subsequence, O(n log n)."""
    tails: list[int] = []
    for x in seq:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _adjacent_pair_accuracy(rank_seq: list[int]) -> float:
    n = len(rank_seq)
    if n < 2:
        return 1.0
    hits = sum(1 for i in range(n - 1) if rank_seq[i + 1] - rank_seq[i] == 1)
    return hits / (n - 1)


def evaluate_roll(rank_seq: list[int], forced_direction: bool = False) -> dict:
    """Score one manuscript given the true rank (0..n-1) of each slip in the
    order a method predicted, trying both reading directions.

    ``rank_seq`` must be a permutation of ``range(len(rank_seq))``.

    ``forced_direction``: skip the reverse-orientation comparison. Use this
    for a manuscript solved with a known anchor (synth_seq.AnchorSpec,
    sequencing.solve_order_cpsat's ``use_anchors``): fixing a title slip as
    first resolves the direction ambiguity that makes "try both, keep the
    better" the right convention everywhere else, so applying that
    convention here would silently credit an anchored roll for a direction
    the anchor evidence had already ruled out.
    """
    n = len(rank_seq)
    if n < 2:
        return dict(kendall_tau=1.0, spearman_rho=1.0,
                    adjacent_pair_acc=1.0, lis_frac=1.0, n=n)

    target = list(range(n))
    candidates = (rank_seq,) if forced_direction else (rank_seq, rank_seq[::-1])
    best = None
    for seq in candidates:
        tau = kendalltau(seq, target).statistic
        rho = spearmanr(seq, target).statistic
        apa = _adjacent_pair_accuracy(seq)
        lis = _lis_length(seq) / n
        score = (tau, rho, apa, lis)
        if best is None or tau > best[0]:
            best = score
    tau, rho, apa, lis = best
    return dict(kendall_tau=float(tau), spearman_rho=float(rho),
                adjacent_pair_acc=float(apa), lis_frac=float(lis), n=n)


def evaluate_batch(df: pd.DataFrame, pred_order: dict[int, list[int]],
                    respect_anchors: bool = False) -> pd.DataFrame:
    """Evaluate a full batch.

    ``df`` has ground truth (``roll_true``, ``order_true``, ``slip_id``).
    ``pred_order`` maps ``roll_true`` -> predicted list of ``slip_id`` in that
    manuscript, in the method's predicted reading order. A roll a method
    failed to recover at all (key absent) scores as chance (see
    ``chance_baseline``) rather than being silently dropped, so a method
    cannot improve its average by giving up on hard manuscripts.

    ``respect_anchors``: if ``df`` has ``is_anchor_first``/``is_anchor_last``
    columns and a roll has any anchored slip, evaluate that roll in its
    forced direction (see ``evaluate_roll``) rather than best-of-two. Only
    meaningful when the solver was actually given ``use_anchors=True``
    (sequencing.solve_order_cpsat) -- passing this without that would score
    a solver that never saw the anchor as if it had committed to a
    direction it had no way to know.
    """
    order_true = dict(zip(df.slip_id, df.order_true))
    has_anchor_col = respect_anchors and "is_anchor_first" in df.columns
    rows = []
    for roll, group in df.groupby("roll_true"):
        n = len(group)
        if roll not in pred_order or len(pred_order[roll]) != n:
            rows.append(dict(roll=roll, **{**chance_baseline(n), "n": n}))
            continue
        rank_seq = [order_true[sid] for sid in pred_order[roll]]
        forced = has_anchor_col and bool(group.is_anchor_first.any())
        rows.append(dict(roll=roll, **evaluate_roll(rank_seq, forced_direction=forced)))
    return pd.DataFrame(rows)


def chance_baseline(n: int, n_trials: int = 200, seed: int = 0) -> dict:
    """Expected score of a uniformly random order, for context in tables."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_trials):
        seq = rng.permutation(n).tolist()
        accs.append(evaluate_roll(seq))
    keys = ["kendall_tau", "spearman_rho", "adjacent_pair_acc", "lis_frac"]
    return {k: float(np.mean([a[k] for a in accs])) for k in keys}


def partition_ari(df: pd.DataFrame, pred_labels: dict[int, int]) -> float:
    """Adjusted Rand index of a predicted manuscript partition against
    ``roll_true``, for the "分册" (split into manuscripts) sub-task."""
    ids = df.slip_id.tolist()
    true = df.roll_true.tolist()
    pred = [pred_labels.get(sid, -1) for sid in ids]
    return float(adjusted_rand_score(true, pred))


def summarize(scored: pd.DataFrame, weight_by_n: bool = True) -> dict:
    """Corpus-level summary of a per-roll score table."""
    if weight_by_n:
        w = scored.n.to_numpy()
    else:
        w = np.ones(len(scored))
    out = {}
    for k in ["kendall_tau", "spearman_rho", "adjacent_pair_acc", "lis_frac"]:
        out[k] = float(np.average(scored[k].to_numpy(), weights=w))
    out["n_rolls"] = len(scored)
    out["n_slips"] = int(scored.n.sum())
    return out
