"""Load a proofread real corpus (CONVENTIONS.md schema) into the same
DataFrame shape ``synth_seq.generate_instance`` produces, so every function
in ``sequencing.py`` / ``metrics_seq.py`` runs on real data completely
unchanged -- no separate "real data path" to keep in sync with the synthetic
one.

Input: one JSON Lines file, one slip per line, matching CONVENTIONS.md's
slip-level schema:

    {"id": "SHD-RS-A-0137", "corpus": "shuihudi_qin", "text_group": "日书甲种",
     "order_gold": 137, "length_mm": 250, "width_mm": 6,
     "notch_positions": [30, 125, 220], "text": "..."}

``length_mm`` and ``notch_positions`` are optional (many editions publish a
corpus-wide length range rather than per-slip measurements): missing values
become NaN, which ``sequencing.channel_scores`` treats as "this channel is
unavailable for this manuscript," not as a crash or a silently-wrong zero.
Per CONVENTIONS.md's 2026-09-03 correction, both are categorisation
evidence, not sequencing evidence (Staack 2016) -- expect them to calibrate
to ~0 weight for ordering, the same as in the synthetic corpus.

``ancient_seqnum`` (a number physically written on the slip, when the
edition's front matter says the manuscript has any) is optional too and is
the one near-ground-truth sequencing signal this schema carries; most real
manuscripts won't have it (Staack 2016).

``text`` is the raw transcription. Turning it into ``text_vec`` -- an
embedding a classical-Chinese language model produces -- is deliberately a
separate step (``embed_fn``), not hard-coded here: the idea document names
several free candidates (Jihuai/bert-ancient-chinese, GuwenBERT, SikuBERT)
and which one to use is a choice to make once there is real text in hand to
try them on, not before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

STRAT_MISSING = -1


def load_jsonl(path: str | Path) -> pd.DataFrame:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return pd.DataFrame(records)


def to_sequencing_frame(raw: pd.DataFrame, embed_fn: Callable[[list[str]], np.ndarray] | None = None,
                         n_cords_default: int = 3) -> pd.DataFrame:
    """Convert a CONVENTIONS.md-schema DataFrame into sequencing.py's format.

    One manuscript = one (corpus, text_group) pair, matching how
    ``synth_seq``'s "roll" is defined. ``order_gold`` becomes ``order_true``
    (0-indexed rank within its manuscript, so ties or gaps in the editors'
    original numbering don't leak into evaluation as if they were data).
    """
    required = {"id", "corpus", "text_group", "order_gold", "text"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = raw.copy()
    df["roll_key"] = list(zip(df.corpus, df.text_group))
    roll_ids = {k: i for i, k in enumerate(sorted(df.roll_key.unique()))}
    df["roll_true"] = df.roll_key.map(roll_ids)
    df["slip_id"] = np.arange(len(df))

    df["order_true"] = (df.groupby("roll_true")["order_gold"]
                         .rank(method="first").astype(int) - 1)

    if "length_mm" not in df.columns:
        df["length_mm"] = np.nan
    if "width_mm" not in df.columns:
        df["width_mm"] = np.nan

    has_notch = "notch_positions" in df.columns and df["notch_positions"].notna().any()
    n_cords = n_cords_default
    if has_notch:
        lens = df["notch_positions"].dropna().apply(len)
        if lens.nunique() > 1:
            raise ValueError(
                "notch_positions has inconsistent lengths across slips "
                f"({sorted(lens.unique())}); pad or split by cord count before loading")
        n_cords = int(lens.iloc[0])
    for j in range(n_cords):
        if has_notch:
            df[f"cord{j}_mm"] = df["notch_positions"].apply(
                lambda v: v[j] if isinstance(v, list) else np.nan)
        else:
            df[f"cord{j}_mm"] = np.nan

    df["strat_level"] = STRAT_MISSING
    if "strat_level" in raw.columns:
        df["strat_level"] = raw["strat_level"].fillna(STRAT_MISSING).astype(int)

    # An ancient number physically written on the slip (簡背/簡側編號), when
    # the edition's front matter says the manuscript has any -- distinct
    # from order_gold, which is the *editors'* assigned order regardless of
    # how they determined it. Most manuscripts don't carry one (Staack 2016).
    df["seqnum"] = STRAT_MISSING
    if "ancient_seqnum" in raw.columns:
        df["seqnum"] = raw["ancient_seqnum"].fillna(STRAT_MISSING).astype(int)

    if embed_fn is not None:
        vecs = embed_fn(df["text"].tolist())
        df["text_vec"] = list(np.asarray(vecs, dtype=np.float32))
    else:
        df["text_vec"] = None  # caller must fill before sequencing.channel_scores runs

    keep = ["slip_id", "roll_true", "order_true", "length_mm", "width_mm",
            *[f"cord{j}_mm" for j in range(n_cords)], "text", "text_vec",
            "strat_level", "seqnum", "id", "corpus", "text_group"]
    return df[keep].reset_index(drop=True)
