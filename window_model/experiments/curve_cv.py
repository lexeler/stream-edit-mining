#!/usr/bin/env python3
"""
Точность в зависимости от длины выпуска, усреднённая по шести слотам.

Нужно понять, достижимы ли 70 % совпадения в принципе. На слоте 2 при
самой короткой нарезке вышло 66,7 %. Если на перекрёстной проверке
кривая заметно выше, значит дело в переносе между эфирами и лечится
данными. Если такая же — нужен новый сигнал.
"""
import sys
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vid+vemb+xv+clap"
WINS = [5.0, 10.0, 20.0, 30.0]
RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def to_grid(p, w, n, w0):
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n) + 0.5) * w0
    return np.interp(g, t, p)


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
            p = np.mean(ps, 0)
            if w == WINS[0]:
                P.setdefault(held, []).append(p)
            else:
                P[held].append(to_grid(p, w, len(P[held][0]), WINS[0]))
        if w == WINS[0]:
            D0 = D
        print(f"окно {w:g} готово", flush=True)

    slots = sorted(P)
    print(f"\n{'доля':>7}" + "".join(f"{s:>8}" for s in slots) + f"{'сред':>9}"
          f"{'случайно':>10}")
    print("-" * (7 + 8 * len(slots) + 19))
    base = np.mean([float((D0[s][0] * D0[s][1]).sum() / D0[s][1].sum())
                    for s in slots])
    for r in RATES:
        row = []
        for s in slots:
            p = np.mean(P[s], 0); Y, WT = D0[s]
            o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
            keep = np.zeros(len(p), bool); keep[o[c <= r]] = True
            ov = (keep * Y * WT).sum(); pred = (keep * WT).sum()
            row.append(ov / pred if pred else 0)
        print(f"{r:>6.0%}" + "".join(f"{v:>8.3f}" for v in row)
              + f"{np.mean(row):>9.3f}{base:>10.3f}")


if __name__ == "__main__":
    main()
