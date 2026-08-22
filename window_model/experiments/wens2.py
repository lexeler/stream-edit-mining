#!/usr/bin/env python3
"""
Развёртка по размеру окна, включая крупные, и их смеси.

В прошлой развёртке лучшим одиночным оказалось окно 30 с, а не 10 —
логично, раз настоящие решения монтажёра идут кусками по минуте и больше.
Проверяем 60 и 120 с и все осмысленные смеси.

Главная цифра — точность при доле 40 %: это и есть «совпадение»,
которое нужно довести до 70 %.
"""
import sys, itertools
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov"
WINS = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0]
BASE = 5.0
RATE = 0.40


def to_grid(p, w, n, w0=BASE):
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n) + 0.5) * w0
    return np.interp(g, t, p)


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def prec_at(p, Y, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    keep = np.zeros(len(p), bool); keep[o[c <= rate]] = True
    ov = (keep * Y * WT).sum(); pred = (keep * WT).sum()
    return float(ov / pred) if pred else 0.0


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    P, D0 = {}, None
    for w in WINS:
        X, D = get_data(w, blocks)
        slots = sorted(X)
        for held in slots:
            tr = [s for s in slots if s != held]
            A = np.vstack([X[s] for s in tr])
            y = np.concatenate([D[s][0] for s in tr])
            wt = np.concatenate([D[s][1] for s in tr])
            ps = []
            for sd in range(3):
                m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=sd,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(A, y, sample_weight=wt)
                ps.append(np.clip(m.predict(X[held]), 0, 1))
            P.setdefault(w, {})[held] = np.mean(ps, 0)
        if w == BASE:
            D0 = D
        print(f"окно {w:g} готово", flush=True)

    slots = sorted(D0)
    n = {s: len(D0[s][0]) for s in slots}
    G = {w: {s: rank01(P[w][s] if w == BASE else to_grid(P[w][s], w, n[s]))
             for s in slots} for w in WINS}

    def measure(ws):
        au, pr = [], []
        for s in slots:
            v = np.mean([G[w][s] for w in ws], 0)
            Y, WT = D0[s]
            au.append(roc_auc_score(Y > 0.5, v))
            pr.append(prec_at(v, Y, WT, RATE))
        return np.mean(au), np.mean(pr)

    print(f"\n{'набор':>26}{'AUC':>8}{'точность при 40 %':>20}")
    print("-" * 54)
    rows = [([w], f"окно {w:g}") for w in WINS]
    rows += [(WINS[:4], "5+10+20+30"), (WINS[2:], "20+30+60+120"),
             (WINS, "все шесть"),
             ([10.0, 30.0, 60.0], "10+30+60"),
             ([5.0, 20.0, 60.0, 120.0], "5+20+60+120")]
    best = None
    for ws, name in rows:
        a, p = measure(ws)
        print(f"{name:>26}{a:>8.3f}{p:>20.3f}")
        if best is None or p > best[1]:
            best = (name, p, a)
    print(f"\nлучшее: {best[0]} — совпадение {best[1]:.3f}, AUC {best[2]:.3f}")


if __name__ == "__main__":
    main()
