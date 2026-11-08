"""Character similarity and document-held-out calibration for text-pair ranking."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def tfidf_embed(texts: list[str]) -> np.ndarray:
    """Character 1--2-gram TF-IDF fitted to the supplied complete corpus."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)
    return vec.fit_transform(texts).toarray()

def zs(S):
    m = ~np.eye(len(S), dtype=bool)
    return (S - S[m].mean()) / (S[m].std() + 1e-9)

def pair_table(T, J, pos):
    """Feature rows for every unordered pair: [zT, zJ], label adjacent."""
    n = len(pos); iu = np.triu_indices(n, 1)
    X = np.stack([zs(T)[iu], zs(J)[iu]], 1)
    y = (np.abs(pos[iu[0]] - pos[iu[1]]) == 1).astype(int)
    return X, y

def calibrate_text_weights(per):
    """Fit weights on other documents. Each entry contains T, J, pos.

    T is cosine similarity; J is the sum of forward and reverse junction
    gains; pos contains reference ranks used only for calibration labels.
    Supply documents in the experiment's deterministic order.
    """
    docs = list(per)
    rng = np.random.default_rng(0)
    feats = {}
    for doc, p in per.items():
        X, y = pair_table(p["T"], p["J"], p["pos"])
        neg = np.where(y == 0)[0]; posi = np.where(y == 1)[0]
        keep = np.concatenate([posi, rng.choice(neg, min(len(neg), 4 * len(posi)), replace=False)])
        feats[doc] = (X[keep], y[keep])
    weights = {}
    for doc in docs:
        X = np.concatenate([feats[o][0] for o in docs if o != doc]); y = np.concatenate([feats[o][1] for o in docs if o != doc])
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        weights[doc] = np.clip(lr.coef_[0], 0, None)

    return weights

def fused_scores(similarity, directional_junction, weights):
    """Score either-side neighbours using calibrated non-negative weights."""
    junction = directional_junction + directional_junction.T
    return weights[0] * zs(similarity) + weights[1] * zs(junction)
