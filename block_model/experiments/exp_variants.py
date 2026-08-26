#!/usr/bin/env python3
"""Перебор вариантов обучения блочной модели по шести обучающим слотам.

Проверяем то, что в прежнем конвейере давало прибавку, но в блочной модели
ещё не проверялось: ранговую нормировку признаков внутри стрима, отдельную
модель для пауз, классификацию вместо регрессии, больше семян.
"""
import os
import numpy as np, sys
from catboost import CatBoostRegressor, CatBoostClassifier

D = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def load(s):
    z = np.load(f"{D}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], z["kind"]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def qnorm(X):
    """ранговая нормировка внутри стрима: у эфиров разные распределения"""
    R = np.argsort(np.argsort(X, 0), 0).astype(np.float32)
    return R / max(len(X) - 1, 1)


def fit(Xtr, ytr, wtr, Xte, seeds=5, cls=False):
    ps = []
    for sd in range(seeds):
        kw = dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=20,
                  verbose=0, random_seed=sd, thread_count=5, rsm=0.3,
                  subsample=0.8, bootstrap_type="Bernoulli")
        if cls:
            m = CatBoostClassifier(**kw)
            m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
            ps.append(m.predict_proba(Xte)[:, 1])
        else:
            m = CatBoostRegressor(**kw)
            m.fit(Xtr, ytr, sample_weight=wtr)
            ps.append(np.clip(m.predict(Xte), 0, 1))
    return np.mean(ps, 0)


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def run(name, variant):
    accs = []
    for held in SLOTS:
        tr = [s for s in SLOTS if s != held]
        X, y, dur, kind = load(held)
        rate = float((dur * y).sum() / dur.sum())
        if variant == "qn":
            Xtr = np.vstack([qnorm(load(s)[0]) for s in tr])
            Xte = qnorm(X)
        else:
            Xtr = np.vstack([load(s)[0] for s in tr])
            Xte = X
        ytr = np.concatenate([load(s)[1] for s in tr])
        wtr = np.concatenate([load(s)[2] for s in tr])
        if variant == "cls":
            p = fit(Xtr, ytr, wtr, Xte, cls=True)
        elif variant == "seeds10":
            p = fit(Xtr, ytr, wtr, Xte, seeds=10)
        elif variant == "split":
            # своя модель для речи и своя для пауз
            ktr = np.concatenate([load(s)[3] for s in tr])
            p = np.zeros(len(y))
            for k in (0, 1):
                m = ktr == k
                sel = kind == k
                if sel.sum() and m.sum():
                    p[sel] = rank01(fit(Xtr[m], ytr[m], wtr[m], Xte[sel]))
        else:
            p = fit(Xtr, ytr, wtr, Xte)
        accs.append(prec(y, dur, p, rate))
        print(f"    слот {held}: {accs[-1]:.3f}", flush=True)
    print(f"  {name}: среднее {np.mean(accs):.4f}", flush=True)
    return np.mean(accs)


if __name__ == "__main__":
    for nm, v in (("как есть", "base"), ("ранговая нормировка", "qn"),
                  ("классификация", "cls"), ("раздельно речь и паузы", "split"),
                  ("десять семян", "seeds10")):
        print(f"### {nm}", flush=True)
        run(nm, v)
