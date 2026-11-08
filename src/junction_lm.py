"""Junction scoring with a classical-Chinese masked LM.

Topical similarity says two slips discuss the same thing; it cannot say
which of five similar slips is the true neighbour. A true neighbour has a
further property: the text runs across the slip boundary, so the last few
characters of A followed by the first few of B should read as language.

For an ordered pair (A, B) we take the last L characters of A and the first
L of B, concatenate them, and score the M characters on each side of the
boundary by masked pseudo-log-likelihood; subtracting the same characters'
pseudo-log-likelihood when only their own slip's window is visible gives a
junction gain: how much the other side of the boundary raises the
probability of these characters. This is directional and independent of
topical n-gram overlap. Model: Jihuai/bert-ancient-chinese (local copy).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import common

MODEL_DIR = common.ROOT / "models" / "bert-ancient-chinese"
L = 8   # window characters per side
M = 2   # scored characters per side of the boundary


def load_mlm(model_dir=MODEL_DIR):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return tok, model


def _masked_logprobs(tok, model, seqs: list[list[str]], targets: list[tuple[int, str]],
                     batch: int = 256) -> np.ndarray:
    """seqs[i] is a list of characters; targets[i] = (position, char). Returns
    log p(char at position | seq with that position masked)."""
    out = np.zeros(len(seqs), dtype=np.float32)
    mask_id, cls, sep = tok.mask_token_id, tok.cls_token_id, tok.sep_token_id
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            chunk = seqs[s:s + batch]; tg = targets[s:s + batch]
            ids_list, pos_list, tgt_ids = [], [], []
            for chars, (p, c) in zip(chunk, tg):
                ids = tok.convert_tokens_to_ids(chars)
                ids = [cls] + ids + [sep]
                ids[p + 1] = mask_id
                ids_list.append(ids); pos_list.append(p + 1)
                tgt_ids.append(tok.convert_tokens_to_ids([c])[0])
            width = max(len(x) for x in ids_list)
            inp = torch.full((len(ids_list), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(inp)
            for i, ids in enumerate(ids_list):
                inp[i, :len(ids)] = torch.tensor(ids); att[i, :len(ids)] = 1
            logits = model(input_ids=inp, attention_mask=att).logits
            lp = torch.log_softmax(logits[torch.arange(len(ids_list)), torch.tensor(pos_list)], -1)
            out[s:s + len(chunk)] = lp[torch.arange(len(ids_list)), torch.tensor(tgt_ids)].numpy()
    return out


def junction_gains(texts: list[str], pairs: list[tuple[int, int]], tok=None, model=None,
                   verbose: bool = True) -> np.ndarray:
    """Gain for each ordered pair (a, b): sum over the M last chars of a's tail
    and the M first chars of b's head of [log p(c | tail+head) - log p(c | own window)]."""
    if tok is None:
        tok, model = load_mlm()
    tails = [list(t[-L:]) for t in texts]
    heads = [list(t[:L]) for t in texts]
    # baselines: each scored char in its own window
    base_seqs, base_tg, base_key = [], [], []
    for i, (tl, hd) in enumerate(zip(tails, heads)):
        for k in range(min(M, len(tl))):
            p = len(tl) - 1 - k
            base_seqs.append(tl); base_tg.append((p, tl[p])); base_key.append(("t", i, k))
        for k in range(min(M, len(hd))):
            base_seqs.append(hd); base_tg.append((k, hd[k])); base_key.append(("h", i, k))
    t0 = time.time()
    base = dict(zip(base_key, _masked_logprobs(tok, model, base_seqs, base_tg)))
    if verbose:
        print(f"[junction] {len(base_seqs)} baseline passes in {time.time() - t0:.0f}s", file=sys.stderr)
    # pair passes
    seqs, tg, key = [], [], []
    for pi, (a, b) in enumerate(pairs):
        tl, hd = tails[a], heads[b]
        win = tl + hd
        for k in range(min(M, len(tl))):
            p = len(tl) - 1 - k
            seqs.append(win); tg.append((p, tl[p])); key.append((pi, ("t", a, k)))
        for k in range(min(M, len(hd))):
            p = len(tl) + k
            seqs.append(win); tg.append((p, hd[k])); key.append((pi, ("h", b, k)))
    t0 = time.time()
    lp = _masked_logprobs(tok, model, seqs, tg)
    if verbose:
        print(f"[junction] {len(seqs)} pair passes in {time.time() - t0:.0f}s "
              f"({len(seqs) / max(1e-9, time.time() - t0):.0f}/s)", file=sys.stderr)
    gain = np.zeros(len(pairs), dtype=np.float32)
    for (pi, bk), v in zip(key, lp):
        gain[pi] += v - base[bk]
    return gain


if __name__ == "__main__":
    tok, model = load_mlm()
    texts = ["昔周武王監觀商王之不禋祀不乃乍帝以祀天神名之曰", "千以尅反商邑尃政天下王=大于周卿李者正萬民弗刃于氒心",
             "乃歸王于龍白和立十又四年王生洹=王=即立白和歸于宋洹"]
    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    g = junction_gains(texts, pairs, tok, model)
    for (a, b), v in zip(pairs, g):
        print(a, b, round(float(v), 2))
    # throughput
    seqs = [list("昔周武王監觀商王之不禋祀不乃乍帝")] * 2048
    t0 = time.time(); _masked_logprobs(tok, model, seqs, [(5, "觀")] * 2048); print(f"{2048 / (time.time() - t0):.0f} passes/s")
