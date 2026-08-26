#!/usr/bin/env python3
"""Вклад описаний кадров: одна и та же модель с ними и без них."""
import os
import numpy as np
from catboost import CatBoostRegressor

D = os.environ.get("SEM_BLOCKS", "work/blocks")
TR = ["1", "3", "6"]
HELD = "7"


def load(s):
    z = np.load(f"{D}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], [str(x) for x in z["names"]]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool)
        k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def fit(Xtr, ytr, wtr, Xte, seeds=4):
    ps = []
    for sd in range(seeds):
        m = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05,
                              l2_leaf_reg=20, verbose=0, random_seed=sd,
                              thread_count=5, rsm=0.3, subsample=0.8,
                              bootstrap_type="Bernoulli")
        m.fit(Xtr, ytr, sample_weight=wtr)
        ps.append(np.clip(m.predict(Xte), 0, 1))
    return np.mean(ps, 0)


X7, y7, d7, names = load(HELD)
rate = float((d7 * y7).sum() / d7.sum())
desc_cols = np.array([n.startswith("desc_") for n in names])
print(f"признаков всего {len(names)}, из них описания кадров {desc_cols.sum()}")
for use in (False, True):
    sel = np.ones(len(names), bool) if use else ~desc_cols
    Xtr = np.vstack([load(s)[0][:, sel] for s in TR])
    ytr = np.concatenate([load(s)[1] for s in TR])
    wtr = np.concatenate([load(s)[2] for s in TR])
    p = fit(Xtr, ytr, wtr, X7[:, sel])
    tag = "с описаниями кадров" if use else "без описаний      "
    print(f"{tag}: точность {prec(y7, d7, p, rate):.3f}")
print(f"случайно {rate:.3f}, обучение на слотах {'+'.join(TR)}")
