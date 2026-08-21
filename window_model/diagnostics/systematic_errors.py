#!/usr/bin/env python3
"""
Где модель ошибается систематически.

Мы упёрлись в потолок наращиванием признаков. Значит надо смотреть не на
средние цифры, а на сами ошибки: чем куски, взятые ЗРЯ, отличаются от взятых
верно. Если найдётся устойчивая разница — это подсказка, чего модели
не хватает, и её можно закрыть прицельно.

Считаем по обучающим слотам (отложенный не трогаем): берём предсказание
модели, отбираем её выпуск, делим взятое на верное и ошибочное
и сравниваем распределения признаков.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, AGG_MEAN, AGG_WORD

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn"
RATE = 0.40
WSEC = 10.0
BASE = ["длина окна", "доля речи", "слов", "слов в минуту", "доля пауз",
        "макс пауза", "пауз>1с", "пауз>3с"]


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(WSEC, blocks)
    slots = sorted(X)
    names = BASE + sum([[k, k + " разброс"] for k in AGG_MEAN], []) + \
        ["слово:" + k for k in AGG_WORD]
    nb = len(names)

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

    # собираем взятое верно и взятое зря по всем слотам
    TP, FP, MISS = [], [], []
    lenTP, lenFP = [], []
    for s in slots:
        p, (Y, WT) = P[s], D[s]
        o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
        keep = np.zeros(len(p), bool); keep[o[c <= RATE]] = True
        good = Y > 0.5
        F = X[s][:, :nb]
        TP.append(F[keep & good]); FP.append(F[keep & ~good])
        MISS.append(F[~keep & good])
        # длины непрерывных полос ошибок
        for mask, acc in ((keep & ~good, lenFP), (keep & good, lenTP)):
            k = 0
            while k < len(mask):
                if mask[k]:
                    j = k
                    while j < len(mask) and mask[j]:
                        j += 1
                    acc.append(WT[k:j].sum()); k = j
                else:
                    k += 1
    TP = np.vstack(TP); FP = np.vstack(FP); MISS = np.vstack(MISS)
    print(f"\nвзято верно {len(TP)} окон, взято зря {len(FP)}, "
          f"упущено {len(MISS)}")
    print(f"полосы: верных медиана {np.median(lenTP):.0f} с, "
          f"ошибочных {np.median(lenFP):.0f} с; "
          f"ошибочных длиннее 60 с: {sum(1 for x in lenFP if x>60)} "
          f"({sum(x for x in lenFP if x>60)/sum(lenFP)*100:.0f} % времени ошибок)")

    print(f"\n{'признак':>22}{'взято верно':>13}{'взято зря':>12}"
          f"{'разница σ':>11}")
    print("-" * 58)
    rows = []
    for j, n in enumerate(names):
        a, b = TP[:, j], FP[:, j]
        sd = np.concatenate([a, b]).std() + 1e-9
        rows.append(((a.mean() - b.mean()) / sd, n, a.mean(), b.mean()))
    for d, n, a, b in sorted(rows, key=lambda z: -abs(z[0]))[:14]:
        print(f"{n:>22}{a:>13.3f}{b:>12.3f}{d:>+11.2f}")

    print(f"\n{'признак':>22}{'упущено':>13}{'взято зря':>12}{'разница σ':>11}")
    print("-" * 58)
    rows = []
    for j, n in enumerate(names):
        a, b = MISS[:, j], FP[:, j]
        sd = np.concatenate([a, b]).std() + 1e-9
        rows.append(((a.mean() - b.mean()) / sd, n, a.mean(), b.mean()))
    for d, n, a, b in sorted(rows, key=lambda z: -abs(z[0]))[:10]:
        print(f"{n:>22}{a:>13.3f}{b:>12.3f}{d:>+11.2f}")


if __name__ == "__main__":
    main()
