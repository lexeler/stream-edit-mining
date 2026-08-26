#!/usr/bin/env python3
"""Отдельная модель для хвоста эфира.

В последних 10 % общая модель даёт 0,438 против 0,713 в началах, а местами
0,00 и 0,21. Возможная причина: в хвосте другие закономерности (застолье,
сборы, прощание), и общая модель их не выучивает, потому что хвост — малая
часть обучающих данных. Проверяем: поможет ли модель, обученная ТОЛЬКО
на хвостах шести эфиров.
"""
import os
import numpy as np
from catboost import CatBoostClassifier

D = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]
TAIL = 0.15


def load(s):
    z = np.load(f"{D}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], z["start"]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def fit(Xtr, ytr, wtr, Xte, seeds=4):
    ps = []
    for sd in range(seeds):
        m = CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05,
                               l2_leaf_reg=25, verbose=0, random_seed=sd,
                               thread_count=5, rsm=0.3, subsample=0.8,
                               bootstrap_type="Bernoulli")
        m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
        ps.append(m.predict_proba(Xte)[:, 1])
    return np.mean(ps, 0)


print(f"{'слот':>5}{'общая':>9}{'хвостовая':>11}{'смесь':>9}{'блоков в обуч.':>16}")
agg = []
for held in SLOTS:
    tr = [s for s in SLOTS if s != held]
    X, y, dur, start = load(held)
    T = float(start[-1] + dur[-1])
    m_te = start >= (1 - TAIL) * T
    rate = float((dur[m_te] * y[m_te]).sum() / dur[m_te].sum())
    # общая модель
    Xtr = np.vstack([load(s)[0] for s in tr])
    ytr = np.concatenate([load(s)[1] for s in tr])
    wtr = np.concatenate([load(s)[2] for s in tr])
    p_all = fit(Xtr, ytr, wtr, X[m_te])
    # хвостовая
    Xs, ys, ws = [], [], []
    for s in tr:
        Xs_, ys_, ds_, st_ = load(s)
        Ts = float(st_[-1] + ds_[-1])
        m = st_ >= (1 - TAIL) * Ts
        Xs.append(Xs_[m]); ys.append(ys_[m]); ws.append(ds_[m])
    p_tail = fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), X[m_te])

    def r01(v):
        r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
        return r / max(len(v) - 1, 1)
    a1 = prec(y[m_te], dur[m_te], r01(p_all), rate)
    a2 = prec(y[m_te], dur[m_te], r01(p_tail), rate)
    a3 = prec(y[m_te], dur[m_te], 0.5 * r01(p_all) + 0.5 * r01(p_tail), rate)
    agg.append((a1, a2, a3))
    print(f"{held:>5}{a1:>9.3f}{a2:>11.3f}{a3:>9.3f}{sum(len(x) for x in Xs):>16}")
a = np.array(agg)
print(f"{'средн':>5}{a[:,0].mean():>9.3f}{a[:,1].mean():>11.3f}{a[:,2].mean():>9.3f}")
