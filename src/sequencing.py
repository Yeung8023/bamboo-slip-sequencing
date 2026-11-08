"""Constrained seriation: recovering the binding order of a manuscript.

The problem is a maximum-weight open Hamiltonian path: given n slips already
known to belong to one manuscript, find the permutation that maximises the
sum of an adjacency-compatibility score over consecutive slips. This is
``yunmeng``'s solving idiom (build a scored graph, hand it to CP-SAT) applied
to a different graph and a different constraint -- a path visiting every node
exactly once, rather than a max-weight set packing over candidate joins.

Reading direction is never recoverable from the evidence alone (see
``metrics_seq.py``), so the solver is free to return either orientation; only
evaluation cares which one was picked.

Candidate arcs are restricted to each slip's k nearest neighbours by combined
score before the model is built (the same practical device LKH and most
large-scale TSP solvers use): an all-pairs circuit model is O(n^2) arcs, which
stops being solvable to a useful gap well before the corpus scales this
project targets (up to 2000 slips), while a sparsified graph keeps the search
tractable and only risks excluding the true neighbour of a slip whose
evidence is extremely noisy -- which is itself informative and is reported as
the "infeasible" rate at the largest scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

CORD_COLS = None  # discovered per-DataFrame, see _cord_cols


def _cord_cols(df: pd.DataFrame) -> list[str]:
    global CORD_COLS
    cols = [c for c in df.columns if c.startswith("cord") and c.endswith("_mm")]
    CORD_COLS = sorted(cols)
    return CORD_COLS


EVIDENCE_CHANNELS = ("notch", "length", "text", "strat", "seqnum")


@dataclass
class EvidenceWeights:
    # notch and length are measured (see channel_scores) but, following
    # Staack (2016)'s criteria catalogue, carry no real order signal within
    # a roll -- they stay in the fusion only so a corpus that violates this
    # (real slips are never perfectly uniform) isn't silently disallowed
    # from using them; a calibration run is expected to weight them near 0.
    notch: float = 1.0
    length: float = 1.0
    text: float = 1.0
    strat: float = 1.0
    seqnum: float = 1.0

    def as_dict(self) -> dict:
        return dict(notch=self.notch, length=self.length, text=self.text,
                    strat=self.strat, seqnum=self.seqnum)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def channel_scores(group: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Per-channel n x n z-scored compatibility terms, before fusion.

    Kept separate from fusion (``combine_scores``) so that
    ``calibrate_weights`` can fit the fusion weights on exactly the features
    ``combine_scores`` will later multiply them by, rather than on some
    other representation.
    """
    ids = group.slip_id.to_numpy()
    n = len(ids)
    cords = group[_cord_cols(group)].to_numpy()
    length = group.length_mm.to_numpy()
    text = np.stack(group.text_vec.to_numpy())
    strat = group.strat_level.to_numpy()
    seqnum = (group.seqnum.to_numpy() if "seqnum" in group.columns
              else np.full(len(group), -1))

    # Real editions do not always report per-slip notch positions or (less
    # often) length; a whole-channel gap is not the same as strat's per-pair
    # gap and is handled separately here, as an all-zero (neutral) channel,
    # so a real corpus missing a channel degrades to "that weight can't be
    # calibrated above 0" instead of poisoning every other channel with NaN.
    if np.isnan(cords).any():
        d_notch = np.zeros((n, n))
    else:
        d_notch = np.linalg.norm(cords[:, None, :] - cords[None, :, :], axis=-1)
    if np.isnan(length).any():
        d_length = np.zeros((n, n))
    else:
        d_length = np.abs(length[:, None] - length[None, :])
    tn = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-9)
    sim_text = tn @ tn.T

    has_strat = strat >= 0
    d_strat = np.abs(strat[:, None].astype(float) - strat[None, :].astype(float))
    strat_ok = has_strat[:, None] & has_strat[None, :]

    # Sequence numbers: present on a minority of manuscripts (SeqNumSpec),
    # missing (-1) elsewhere on a per-slip basis just like strat_level, so
    # handled the same way -- a pair-level mask, not a whole-channel gap.
    has_seqnum = seqnum >= 0
    d_seqnum = np.abs(seqnum[:, None].astype(float) - seqnum[None, :].astype(float))
    seqnum_ok = has_seqnum[:, None] & has_seqnum[None, :]

    s_notch = -_zscore(d_notch.reshape(-1)).reshape(n, n)
    s_length = -_zscore(d_length.reshape(-1)).reshape(n, n)
    s_text = _zscore(sim_text.reshape(-1)).reshape(n, n)
    s_strat = np.zeros((n, n))
    if strat_ok.any():
        vals = -_zscore(d_strat[strat_ok])
        s_strat[strat_ok] = vals
    s_seqnum = np.zeros((n, n))
    if seqnum_ok.any():
        vals = -_zscore(d_seqnum[seqnum_ok])
        s_seqnum[seqnum_ok] = vals

    channels = dict(notch=s_notch, length=s_length, text=s_text,
                     strat=s_strat, seqnum=s_seqnum)
    coverage_ok = dict(strat=strat_ok, seqnum=seqnum_ok)
    return channels, coverage_ok, ids


def combine_scores(channels: dict[str, np.ndarray], coverage_ok: dict[str, np.ndarray],
                    weights: EvidenceWeights) -> np.ndarray:
    """Fuse per-channel scores into one adjacency-compatibility matrix.

    Setting a weight to 0 removes a channel from the ablation cleanly,
    without changing anything else about the model.
    """
    w = weights.as_dict()
    score = (w["notch"] * channels["notch"] + w["length"] * channels["length"]
             + w["text"] * channels["text"]
             + w["strat"] * channels["strat"] * coverage_ok["strat"]
             + w["seqnum"] * channels["seqnum"] * coverage_ok["seqnum"])
    np.fill_diagonal(score, -np.inf)
    return score


def pairwise_scores(group: pd.DataFrame, weights: EvidenceWeights) -> tuple[np.ndarray, np.ndarray]:
    """Dense n x n adjacency-compatibility matrix for one manuscript."""
    channels, coverage_ok, ids = channel_scores(group)
    return combine_scores(channels, coverage_ok, weights), ids


def calibrate_weights(df_calib: pd.DataFrame, seed: int = 0,
                       neg_per_pos: int = 4) -> EvidenceWeights:
    """Fit fusion weights on a held-out calibration corpus, the way
    ``yunmeng``'s ``assemble.py`` fits ``fit_context_lr`` on a calibration
    split rather than hand-picking weights: a logistic regression predicting
    "are these two slips true neighbours" from the four z-scored channel
    terms, so a channel that is individually strong but only weakly
    *independent* of the others (or one that is simply noisier in this
    corpus) is not given the same say as one that is both strong and
    informative on its own.

    Coefficients are clipped at 0: a learned negative weight would mean
    "prefer slips whose evidence looks like non-neighbours," which is a
    calibration artefact on a finite sample, not a real inverse
    relationship, and would make the ablation (weight -> 0 removes a
    channel) ambiguous with "weight -> negative helps."
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    order = ["notch", "length", "text", "strat", "seqnum"]
    X, y = [], []
    for roll, group in df_calib.groupby("roll_true"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 3:
            continue
        channels, coverage_ok, ids = channel_scores(group)
        order_true = group.order_true.to_numpy()
        adj = np.abs(order_true[:, None] - order_true[None, :]) == 1
        iu = np.triu_indices(n, k=1)
        pos_mask = adj[iu]
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(~pos_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        take = min(len(neg_idx), max(1, len(pos_idx) * neg_per_pos))
        neg_sample = rng.choice(neg_idx, size=take, replace=False)
        keep = np.concatenate([pos_idx, neg_sample])
        rows_i, cols_j = iu[0][keep], iu[1][keep]
        feats = np.stack([channels[c][rows_i, cols_j] for c in order], axis=1)
        labels = pos_mask[keep].astype(int)
        X.append(feats)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    coef = np.clip(clf.coef_.ravel(), 0.0, None)
    if coef.sum() < 1e-9:
        coef = np.ones(len(order))  # degenerate fit: fall back to equal weights
    return EvidenceWeights(**dict(zip(order, coef.tolist())))


def solve_order_cpsat(group: pd.DataFrame, weights: EvidenceWeights,
                       k_neighbors: int = 14, time_limit_s: float = 20.0,
                       seed: int = 0, use_anchors: bool = False,
                       strat_hard_window: float | None = None,
                       score_matrix: np.ndarray | None = None) -> tuple[list[int], dict]:
    """Solve for the best reading order of one manuscript with CP-SAT.

    Two *hard* constraints, on top of the soft evidence fusion in
    ``pairwise_scores`` -- idea-13.md's "篇章边界" and a stricter form of its
    "层位相容" criterion. The other three constraints idea-13.md lists
    (契口对齐 / 简长一致 / 卷收几何) are not implemented: per synth_seq.RollSpec's
    2026-09-04 correction, the first two are categorisation evidence and the
    third has no literature support, so encoding any of them as a *sequencing*
    constraint would just reintroduce the mistake that correction fixed.

    ``use_anchors``: if the manuscript has a known first/last slip
    (``is_anchor_first``/``is_anchor_last`` columns, see synth_seq.AnchorSpec
    -- a title slip or unambiguous opening/closing, a real philological
    practice, not fabricated), fix it as such rather than leaving direction
    free. This also resolves the direction ambiguity ``metrics_seq.py``
    otherwise handles by taking the better of two orientations: an anchored
    roll's direction is no longer ambiguous, and evaluation should not use
    that convention for it (see exp_constraints.py).

    ``strat_hard_window``: if set, forbid a candidate arc between two slips
    whose recorded stratigraphic levels differ by more than this many
    levels (when both are known) -- turning the soft strat score into an
    actual hard constraint, rather than just a preference. Slips with no
    stratigraphic record are unaffected (nothing to enforce).
    """
    if score_matrix is not None:
        # Caller-supplied arc weights, e.g. a directional junction score
        # (junction_lm.py). Arcs take score[i, j] individually, so an
        # asymmetric matrix makes the path direction-aware with no other
        # change to the model.
        score, ids = np.asarray(score_matrix, dtype=float), group.slip_id.to_numpy()
    else:
        score, ids = pairwise_scores(group, weights)
    n = len(ids)
    if n <= 2:
        return ids.tolist(), dict(status="TRIVIAL", objective=0.0, gap=0.0, wall_s=0.0)

    k = min(k_neighbors, n - 1)
    scale = 1000.0

    # Candidate arcs are each node's top-k neighbours in *either* direction:
    # a plain per-row top-k is not symmetric (j can be i's nearest neighbour
    # without i being j's), and AddCircuit needs every node to have viable
    # incoming as well as outgoing arcs or the model is spuriously
    # infeasible.
    pair_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argsort(-score[i])[:k]:
            if j == i:
                continue
            pair_set.add((int(i), int(j)))
            pair_set.add((int(j), int(i)))

    if strat_hard_window is not None:
        strat = group.strat_level.to_numpy()
        has_strat = strat >= 0
        pair_set = {
            (i, j) for (i, j) in pair_set
            if not (has_strat[i] and has_strat[j])
            or abs(int(strat[i]) - int(strat[j])) <= strat_hard_window
        }

    anchor_first_pos = anchor_last_pos = None
    if use_anchors and "is_anchor_first" in group.columns:
        first_hits = np.where(group.is_anchor_first.to_numpy())[0]
        last_hits = np.where(group.is_anchor_last.to_numpy())[0]
        if len(first_hits):
            anchor_first_pos = int(first_hits[0])
        if len(last_hits):
            anchor_last_pos = int(last_hits[0])

    model = cp_model.CpModel()
    DEPOT = n
    arcs = []
    lits = {}

    for i, j in pair_set:
        lit = model.NewBoolVar(f"a_{i}_{j}")
        lits[(i, j)] = lit
        arcs.append((i, j, lit))
    depot_in, depot_out = {}, {}  # DEPOT -> i (i is first) / i -> DEPOT (i is last)
    for i in range(n):
        d_in = model.NewBoolVar(f"d_in_{i}")
        d_out = model.NewBoolVar(f"d_out_{i}")
        arcs.append((DEPOT, i, d_in))
        arcs.append((i, DEPOT, d_out))
        depot_in[i] = d_in
        depot_out[i] = d_out
    model.AddCircuit(arcs)

    if anchor_first_pos is not None:
        model.Add(depot_in[anchor_first_pos] == 1)
    if anchor_last_pos is not None:
        model.Add(depot_out[anchor_last_pos] == 1)

    obj = []
    for (i, j), lit in lits.items():
        obj.append(int(round(score[i, j] * scale)) * lit)
    model.Maximize(sum(obj))

    # Warm-start from the greedy chain: at larger n a sparsified circuit can
    # take a long time to find *any* feasible solution before it starts
    # improving one, which without a hint can leave the solver with nothing
    # to report at all inside the time limit (see exp_scaling.py's n~1800
    # runs, UNKNOWN status). A hint only needs to cover a subset of
    # variables -- CP-SAT completes the rest by search -- so it's safe even
    # where the greedy chain used an edge outside this candidate graph.
    # This ortools build's add_hint takes one (var, value) pair per call,
    # not the list form the API docs show for other versions.
    greedy_chain = _greedy_positions(score, n)
    for a, b in zip(greedy_chain[:-1], greedy_chain[1:]):
        if (a, b) in lits:
            model.add_hint(lits[(a, b)], 1)
    model.add_hint(depot_in[greedy_chain[0]], 1)
    model.add_hint(depot_out[greedy_chain[-1]], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = seed
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # At large n the search can exhaust the time budget without ever
        # confirming a feasible circuit even though the hint above gave it
        # one to start from (observed at n~1800: exp_scaling.py, UNKNOWN).
        # Falling back to the arbitrary input order in that case reports a
        # solver failure as if it were a modelling one -- worse than the
        # greedy baseline we already computed for the hint, which is never
        # the right thing for a "give me your best answer" API to return.
        fallback = [int(ids[i]) for i in greedy_chain]
        return fallback, dict(status=status_name, objective=float("nan"),
                               gap=float("nan"), wall_s=solver.WallTime())

    succ = {}
    for (i, j), lit in lits.items():
        if solver.Value(lit):
            succ[i] = j
    start = next(i for i, lit in depot_in.items() if solver.Value(lit))
    order_idx = [start]
    seen = {start}
    cur = succ.get(start)
    while cur is not None and cur not in seen:
        seen.add(cur)
        order_idx.append(cur)
        cur = succ.get(cur)

    gap = 0.0
    try:
        best_bound = solver.BestObjectiveBound()
        obj_val = solver.ObjectiveValue()
        if abs(obj_val) > 1e-9:
            gap = abs(best_bound - obj_val) / max(1e-9, abs(obj_val))
    except Exception:
        pass

    pred_ids = [int(ids[i]) for i in order_idx]
    return pred_ids, dict(status=status_name, objective=solver.ObjectiveValue(),
                           gap=gap, wall_s=solver.WallTime())


def _greedy_positions(score: np.ndarray, n: int, seed: int = 0) -> list[int]:
    """Nearest-neighbour chaining on a raw score matrix, by position (not
    slip_id). Shared by ``solve_order_greedy`` (the baseline) and
    ``solve_order_cpsat`` (as a warm-start hint, see there for why)."""
    if n <= 2:
        return list(range(n))

    rng = np.random.default_rng(seed)
    i0, j0 = np.unravel_index(np.argmax(score), score.shape)
    chain = [int(i0), int(j0)]
    used = set(chain)

    while len(chain) < n:
        head, tail = chain[0], chain[-1]
        best = None
        for end, node in (("head", head), ("tail", tail)):
            cand = [(score[node, m], m) for m in range(n) if m not in used]
            if not cand:
                continue
            s, m = max(cand, key=lambda t: t[0])
            if best is None or s > best[0]:
                best = (s, m, end)
        if best is None:
            remaining = [m for m in range(n) if m not in used]
            m = rng.choice(remaining)
            chain.append(int(m))
            used.add(int(m))
            continue
        _, m, end = best
        if end == "head":
            chain.insert(0, m)
        else:
            chain.append(m)
        used.add(m)

    return chain


def solve_order_greedy(group: pd.DataFrame, weights: EvidenceWeights,
                        seed: int = 0) -> list[int]:
    """Nearest-neighbour chaining: start from the pair with the single
    highest compatibility score, then repeatedly extend whichever end of the
    growing chain has the best-scoring unused neighbour.

    This is the sequencing analogue of yunmeng's ``mutual_best`` heuristic
    (baselines.py): a locally greedy stand-in for the globally optimal
    solver, useful to show how much the optimisation step itself is worth.
    """
    score, ids = pairwise_scores(group, weights)
    n = len(ids)
    chain = _greedy_positions(score, n, seed=seed)
    return [int(ids[i]) for i in chain]


def solve_order_spectral(group: pd.DataFrame, weights: EvidenceWeights,
                          score_matrix: np.ndarray | None = None) -> list[int]:
    """Spectral seriation: order slips by the Fiedler vector of the Laplacian
    of the similarity matrix (Atkins, Boman & Hendrickson 1998).

    This is the standard baseline of the seriation literature and the natural
    competitor to the constrained path: where CP-SAT maximises the summed
    score of consecutive pairs under hard constraints, the spectral method
    relaxes the ordering to a continuous embedding and reads the order off
    one eigenvector. It is included because a reader coming from seriation
    would ask for it, and because it costs one eigendecomposition rather than
    a search.

    Similarities must be non-negative for the Laplacian to behave, so the
    fused score (which is z-scored and therefore signed) is shifted to zero
    minimum before the Laplacian is formed.
    """
    if score_matrix is not None:
        score, ids = np.asarray(score_matrix, dtype=float), group.slip_id.to_numpy()
    else:
        score, ids = pairwise_scores(group, weights)
    n = len(ids)
    if n <= 2:
        return ids.tolist()

    S = (score + score.T) / 2.0
    np.fill_diagonal(S, 0.0)
    S = S - S.min()
    np.fill_diagonal(S, 0.0)
    d = S.sum(axis=1)
    L = np.diag(d) - S
    try:
        vals, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return ids.tolist()
    order_idx = np.argsort(vecs[:, 1])       # Fiedler vector
    return [int(ids[i]) for i in order_idx]


def solve_order_random(group: pd.DataFrame, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = group.slip_id.to_numpy().copy()
    rng.shuffle(ids)
    return ids.tolist()


def solve_batch(df: pd.DataFrame, method: str, weights: EvidenceWeights | None = None,
                 seed: int = 0, **kw) -> tuple[dict[int, list[int]], pd.DataFrame]:
    """Solve every manuscript in a mixed batch. Assumes ``roll_true`` is
    known (partition already resolved -- see ``partition.py`` -- so this
    module's contribution is the ordering step, matching the idea document's
    two-stage design: divide into manuscripts, then order within one)."""
    weights = weights or EvidenceWeights()
    pred = {}
    logs = []
    for roll, group in df.groupby("roll_true"):
        group = group.reset_index(drop=True)
        if method == "random":
            order = solve_order_random(group, seed=seed + int(roll))
            logs.append(dict(roll=roll, status="RANDOM"))
        elif method == "greedy":
            order = solve_order_greedy(group, weights, seed=seed + int(roll))
            logs.append(dict(roll=roll, status="GREEDY"))
        elif method == "spectral":
            order = solve_order_spectral(group, weights, **kw)
            logs.append(dict(roll=roll, status="SPECTRAL"))
        elif method == "cpsat":
            order, info = solve_order_cpsat(group, weights, seed=seed + int(roll), **kw)
            logs.append(dict(roll=roll, **info))
        else:
            raise ValueError(method)
        pred[roll] = order
    return pred, pd.DataFrame(logs)
