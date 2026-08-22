#!/usr/bin/env python3
"""
Решать сценами, а не окнами.

Разбор ошибок показал системный перекос: модель предпочитает плотную
быструю речь и упускает тихие места (плотность речи у упущенного 0,397
против 0,481 у взятого зря, пауз длиннее 3 с втрое больше). Причина
не в признаках, а в способе отбора: мы ранжируем ОКНА и выклёвываем
самые говорливые, тогда как монтажёр оставляет сцену целиком, вместе
с паузами внутри неё.

Здесь: режем эфир на сцены по длинным паузам, оценка сцены — среднее
по её окнам, отбираем сцены целиком. Сравниваем с отбором по окнам.
"""
import sys
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn"
WSEC = 10.0
RATE = 0.40


def prec(keep, Y, WT):
    ov = float((keep * Y * WT).sum()); pd = float((keep * WT).sum())
    return ov / pd if pd else 0.0, pd


def by_windows(p, Y, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    k = np.zeros(len(p), bool); k[o[c <= rate]] = True
    return k


def scenes_from_gap(F, nac, minlen, wsec):
    """границы сцен там, где длинная пауза; сцена не короче minlen"""
    maxpause = F[:, 5]                      # признак «макс пауза» в окне
    thr = np.quantile(maxpause, 0.80)
    cut = maxpause >= thr
    bounds, last = [0], 0
    for i in range(1, len(F)):
        if cut[i] and (i - last) * wsec >= minlen:
            bounds.append(i); last = i
    bounds.append(len(F))
    return bounds


def by_scenes(p, Y, WT, rate, bounds):
    sc = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = WT[a:b].sum()
        sc.append(((p[a:b] * WT[a:b]).sum() / max(w, 1e-9), w, a, b))
    sc.sort(key=lambda z: -z[0])
    keep = np.zeros(len(p), bool)
    tot, lim = 0.0, rate * WT.sum()
    for s, w, a, b in sc:
        if tot + w <= lim:
            keep[a:b] = True; tot += w
    return keep


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(WSEC, blocks)
    slots = sorted(X)
    # базовые признаки лежат первыми, «макс пауза» — шестой столбец
    P = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        ps = []
        for sd in range(3):
            m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                  l2_leaf_reg=20, verbose=0, random_seed=sd,
                                  thread_count=5, rsm=0.3, subsample=0.8,
                                  bootstrap_type="Bernoulli")
            m.fit(np.vstack([X[s] for s in tr]),
                  np.concatenate([D[s][0] for s in tr]),
                  sample_weight=np.concatenate([D[s][1] for s in tr]))
            ps.append(np.clip(m.predict(X[held]), 0, 1))
        P[held] = np.mean(ps, 0)
        print(f"слот {held} посчитан", flush=True)

    print(f"\n{'способ отбора':>28}{'совпадение':>12}{'минут':>9}")
    print("-" * 50)
    rows = [("по окнам (как сейчас)", None)]
    for ml in (20, 40, 60, 90, 120, 180):
        rows.append((f"сценами, минимум {ml} с", ml))
    for name, ml in rows:
        acc, mins = [], []
        for s in slots:
            p, (Y, WT) = P[s], D[s]
            if ml is None:
                k = by_windows(p, Y, WT, RATE)
            else:
                b = scenes_from_gap(X[s], 45, ml, WSEC)
                k = by_scenes(p, Y, WT, RATE, b)
            q, pd = prec(k, Y, WT)
            acc.append(q); mins.append(pd / 60)
        print(f"{name:>28}{np.mean(acc):>12.3f}{np.mean(mins):>9.0f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
