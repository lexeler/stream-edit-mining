#!/usr/bin/env python3
"""
Что именно предсказывать. Цель «доля окна» смешивает две разные вещи:
насколько кусок хорош сам по себе и сколько монтажёр вообще оставил
в этом эфире (от 26 % до 47 %). Второе для нас шум — долю мы всё равно
задаём отдельно.

Пробуем:
  сырая        доля окна как есть
  центр        минус средняя доля своего стрима (учим «лучше или хуже
               среднего по этому эфиру», а не абсолютный уровень)
  ранг         ранг внутри стрима, 0..1
  сглаж        доля, усреднённая по ±30 с (упор на макро-структуру)

Оценка одинаковая: доля выпуска — смесь средней предсказанной и типичного
хронометража канала.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, score, take_rate

BLOCKS = "ac+pca+tag+vid+xv+clap"


def smooth(y, wt, sec=30.0, wsec=10.0):
    k = max(int(sec / wsec), 1)
    c = np.concatenate([[0.0], np.cumsum(y)])
    lo = np.clip(np.arange(len(y)) - k, 0, len(y))
    hi = np.clip(np.arange(len(y)) + k + 1, 0, len(y))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
    T = {s: float(D[s][1].sum()) for s in slots}
    true_min = {s: float((D[s][0] * D[s][1]).sum()) / 60 for s in slots}

    targets = {
        "сырая": {s: D[s][0] for s in slots},
        "центр": {s: D[s][0] - float((D[s][0] * D[s][1]).sum() / D[s][1].sum())
                  for s in slots},
        "ранг": {s: rank01(D[s][0] + 1e-6 * np.arange(len(D[s][0]))) for s in slots},
        "сглаж": {s: smooth(D[s][0], D[s][1]) for s in slots},
    }

    print(f"{'цель':>8}{'AUC':>8}{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}")
    print("-" * 48)
    for name, Yt in targets.items():
        acc, aucs = [], []
        for held in slots:
            tr = [s for s in slots if s != held]
            ps = []
            for sd in range(3):
                m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=sd,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(np.vstack([X[s] for s in tr]),
                      np.concatenate([Yt[s] for s in tr]),
                      sample_weight=np.concatenate([D[s][1] for s in tr]))
                ps.append(m.predict(X[held]))
            p = np.mean(ps, 0)
            Y, WT = D[held]
            typ = float(np.median([true_min[s] for s in tr]))
            pc = np.clip(p, 0, 1)
            r_mean = float((pc * WT).sum() / WT.sum())
            if name in ("центр", "ранг"):
                r_mean = float(np.mean([true_min[s] * 60 / T[s] for s in tr]))
            r = (r_mean + min(typ * 60 / T[held], 0.85)) / 2
            acc.append(score(take_rate(p, WT, r), Y, WT))
            aucs.append(roc_auc_score(Y > 0.5, p))
        m = np.mean(acc, 0)
        print(f"{name:>8}{np.mean(aucs):>8.3f}{m[0]:>8.3f}{m[1]:>8.3f}"
              f"{m[2]:>8.3f}{m[3]:>8.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
