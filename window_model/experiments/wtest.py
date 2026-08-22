#!/usr/bin/env python3
"""
Вес примеров. Сейчас каждое окно весит свою длительность, то есть
десятиминутный выброшенный кусок даёт 60 почти одинаковых примеров
и перевешивает шестьдесят разных решений в другом месте.

А настоящих решений монтажёра всего около 25 на стрим. Пробуем уравнять:
вес окна делим на длину непрерывного участка, в котором оно лежит.

  длит       вес = длительность окна (как было)
  корень     вес = 1/sqrt(длина участка)
  поровну    вес = 1/длина участка, каждое решение весит одинаково
  край       вес выше у окон рядом с границей решения
"""
import sys
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, score, take_rate

BLOCKS = "ac+pca+tag+vid+xv+clap"


def runlen(y):
    """для каждого окна — длина непрерывного участка одного решения"""
    k = y > 0.5
    out = np.zeros(len(y))
    i = 0
    while i < len(k):
        j = i
        while j < len(k) and k[j] == k[i]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def edge_dist(y):
    k = (y > 0.5).astype(int)
    ch = np.flatnonzero(np.diff(k) != 0)
    if len(ch) == 0:
        return np.full(len(y), len(y), float)
    idx = np.arange(len(y))
    return np.min(np.abs(idx[:, None] - ch[None, :]), 1).astype(float)


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
    T = {s: float(D[s][1].sum()) for s in slots}
    tmin = {s: float((D[s][0] * D[s][1]).sum()) / 60 for s in slots}

    schemes = {
        "длит": lambda s: D[s][1],
        "корень": lambda s: D[s][1] / np.sqrt(runlen(D[s][0])),
        "поровну": lambda s: D[s][1] / runlen(D[s][0]),
        "край": lambda s: D[s][1] / (1 + 0.2 * edge_dist(D[s][0])),
    }
    print(f"{'вес':>10}{'AUC':>8}{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}")
    print("-" * 50)
    for name, fn in schemes.items():
        acc, aucs = [], []
        for held in slots:
            tr = [s for s in slots if s != held]
            A = np.vstack([X[s] for s in tr])
            y = np.concatenate([D[s][0] for s in tr])
            w = np.concatenate([fn(s) for s in tr])
            w = w / w.mean()
            ps = []
            for sd in range(3):
                m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=sd,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(A, y, sample_weight=w)
                ps.append(np.clip(m.predict(X[held]), 0, 1))
            p = np.mean(ps, 0)
            Y, WT = D[held]
            typ = float(np.median([tmin[s] for s in tr]))
            r = (float((p * WT).sum() / WT.sum()) + min(typ * 60 / T[held], 0.85)) / 2
            acc.append(score(take_rate(p, WT, r), Y, WT))
            aucs.append(roc_auc_score(Y > 0.5, p))
        m = np.mean(acc, 0)
        print(f"{name:>10}{np.mean(aucs):>8.3f}{m[0]:>8.3f}{m[1]:>8.3f}"
              f"{m[2]:>8.3f}{m[3]:>8.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
