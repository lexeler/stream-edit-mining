#!/usr/bin/env python3
"""
Усреднение по размеру окна. Модель на 5 с видит подробности, на 30 с —
общий ход сцены. Ошибки у них разные, среднее обычно лучше любой одной.

Оценка на общей сетке 10 с. Доля выпуска — смесь «средней доли» модели
и типичного хронометража канала (проверено ratetest.py).
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, score, take_rate

BLOCKS = "ac+pca+tag+vid+xv+clap"
WINS = [5.0, 10.0, 20.0, 30.0]


def predict(w, blocks, seeds=3):
    X, D = get_data(w, blocks)
    slots = sorted(X)
    P = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        ps = []
        for sd in range(seeds):
            m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                  l2_leaf_reg=20, verbose=0, random_seed=sd,
                                  thread_count=5, rsm=0.3, subsample=0.8,
                                  bootstrap_type="Bernoulli")
            m.fit(np.vstack([X[s] for s in tr]),
                  np.concatenate([D[s][0] for s in tr]),
                  sample_weight=np.concatenate([D[s][1] for s in tr]))
            ps.append(np.clip(m.predict(X[held]), 0, 1))
        P[held] = np.mean(ps, 0)
    return P, D


def to_grid(p, w, n10):
    """перекладываем предсказания с сетки w на сетку 10 с"""
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n10) + 0.5) * 10.0
    return np.interp(g, t, p)


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    base, D10 = predict(10.0, blocks)
    slots = sorted(D10)
    T = {s: float(D10[s][1].sum()) for s in slots}
    true_min = {s: float((D10[s][0] * D10[s][1]).sum()) / 60 for s in slots}

    allp = {10.0: base}
    for w in WINS:
        if w == 10.0:
            continue
        P, _ = predict(w, blocks)
        allp[w] = {s: to_grid(P[s], w, len(D10[s][0])) for s in slots}
        print(f"окно {w:g} посчитано", flush=True)

    def rate_for(held, p):
        tr = [s for s in slots if s != held]
        typ = float(np.median([true_min[s] for s in tr]))
        WT = D10[held][1]
        r_mean = float((p * WT).sum() / WT.sum())
        r_dur = min(typ * 60 / T[held], 0.85)
        return (r_mean + r_dur) / 2

    print(f"\n{'набор':>22}{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}")
    print("-" * 54)
    combos = [("окно 10", [10.0])] + \
             [(f"окно {w:g}", [w]) for w in WINS if w != 10.0] + \
             [("10+20", [10.0, 20.0]), ("5+10+20", [5.0, 10.0, 20.0]),
              ("все четыре", WINS)]
    for name, ws in combos:
        acc = []
        for held in slots:
            p = np.mean([allp[w][held] for w in ws], 0)
            Y, WT = D10[held]
            acc.append(score(take_rate(p, WT, rate_for(held, p)), Y, WT))
        m = np.mean(acc, 0)
        print(f"{name:>22}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>8.3f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
