#!/usr/bin/env python3
"""Смесь классификации и регрессии: два взгляда на одну задачу.

Классификация учится на «войдёт или нет», регрессия — на доле блока,
вошедшей в выпуск. Ошибки у них разные, поэтому смесь может дать больше
каждой по отдельности. Проверка — по шести обучающим слотам.
"""
import os
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier

D = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def load(s):
    z = np.load(f"{D}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


kw = dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=20,
          verbose=0, thread_count=5, rsm=0.3, subsample=0.8,
          bootstrap_type="Bernoulli")
res = {w: [] for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
for held in SLOTS:
    tr = [s for s in SLOTS if s != held]
    Xtr = np.vstack([load(s)[0] for s in tr])
    ytr = np.concatenate([load(s)[1] for s in tr])
    wtr = np.concatenate([load(s)[2] for s in tr])
    X, y, dur = load(held)
    rate = float((dur * y).sum() / dur.sum())
    pc, pr = [], []
    for sd in range(4):
        m = CatBoostClassifier(random_seed=sd, **kw)
        m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
        pc.append(m.predict_proba(X)[:, 1])
        m = CatBoostRegressor(random_seed=sd, **kw)
        m.fit(Xtr, ytr, sample_weight=wtr)
        pr.append(np.clip(m.predict(X), 0, 1))
    c, r = rank01(np.mean(pc, 0)), rank01(np.mean(pr, 0))
    for w in res:
        res[w].append(prec(y, dur, (1 - w) * c + w * r, rate))
    print(f"слот {held} готов", flush=True)
print()
print(f"{'вес регрессии':>15}{'точность':>10}")
for w, v in res.items():
    print(f"{w:>15.2f}{np.mean(v):>10.4f}")
