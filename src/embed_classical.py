"""Classical-Chinese BERT embeddings for the Baoshan original-script corpus.

The paper's Discussion names this as its most direct open question: is the
TF-IDF result on original Chu script near the ceiling, or would a model
trained on classical Chinese find coherence that character n-grams miss?
text2vec-base-chinese (used for the Guodian *translations*) is the wrong
tool here -- it was trained on modern Chinese -- so a BERT pretrained on
pre-modern text is used instead, with mean pooling over the last hidden
state as the sentence vector (these are masked-LM checkpoints, not
sentence-embedding models, so pooling is the standard recipe).

Characters the model's vocabulary does not cover become [UNK]; the fraction
is reported, because a corpus full of rare Chu graphs could defeat a
vocabulary built from transmitted texts, and that would be a finding about
the model, not about coherence.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import common

MODEL_DIR = common.ROOT / "models" / "bert-ancient-chinese"


def load(model_dir=MODEL_DIR):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    return tok, model


def embed(texts: list[str], model_dir=MODEL_DIR, batch: int = 32) -> np.ndarray:
    tok, model = load(model_dir)
    unk_id = tok.unk_token_id
    out, n_tok, n_unk = [], 0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            n_tok += int(mask.sum())
            n_unk += int(((ids == unk_id) & (mask == 1)).sum())
            h = model(**enc).last_hidden_state
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.numpy())
    vec = np.concatenate(out, 0)
    print(f"[embed_classical] {len(texts)} texts, {n_tok} tokens, "
          f"{n_unk} [UNK] ({100 * n_unk / max(1, n_tok):.1f}%)", file=sys.stderr)
    return vec


def whiten(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """BERT-whitening (Su et al. 2021): centre and decorrelate the pooled
    vectors so cosine similarity is not dominated by the few directions a
    raw masked-LM checkpoint pushes every sentence towards. The standard
    cheap remedy for anisotropy, applied here so that a loss to TF-IDF
    cannot be blamed on the pooling alone."""
    mu = x.mean(0, keepdims=True)
    cov = np.cov((x - mu).T)
    u, s, _ = np.linalg.svd(cov)
    w = u @ np.diag(1.0 / np.sqrt(s + eps))
    return (x - mu) @ w


def classical_embed(texts: list[str]) -> np.ndarray:
    """Signature-compatible with the embed_fn expected by load_real_corpus."""
    return embed(texts)


def classical_embed_whitened(texts: list[str]) -> np.ndarray:
    return whiten(embed(texts))
