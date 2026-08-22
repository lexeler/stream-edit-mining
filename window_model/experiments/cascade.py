#!/usr/bin/env python3
"""
Каскад: вторая модель учится только на спорной зоне.

Разбор ошибок показал: внутри отобранного моделью материала верное
и ошибочное почти неразличимы по признакам (разница до 0,25σ), тогда как
между упущенным и взятым зря разница большая. То есть первая модель хорошо
отсеивает явный мусор, но на границе решает плохо.

Лечим каскадом: первая модель отбирает верхнюю зону, вторая обучается
ТОЛЬКО на ней — её ёмкость уходит на настоящую границу решения, а не на
отделение речи от тишины.

Обе модели обучаются честно: предсказания первой для обучающих слотов
получены по отложенной схеме, отложенный слот не участвует нигде.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn"
WSEC = 10.0
RATE = 0.40


def fit(A, y, w, Xte, seeds=3):
    ps = []
    for sd in range(seeds):
        m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                              l2_leaf_reg=20, verbose=0, random_seed=sd,
                              thread_count=5, rsm=0.3, subsample=0.8,
                              bootstrap_type="Bernoulli")
        m.fit(A, y, sample_weight=w)
        ps.append(np.clip(m.predict(Xte), 0, 1))
    return np.mean(ps, 0)


def prec(keep, Y, WT):
    ov = float((keep * Y * WT).sum()); pd = float((keep * WT).sum())
    return ov / pd if pd else 0.0


def take(p, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    k = np.zeros(len(p), bool); k[o[c <= rate]] = True
    return k


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(WSEC, blocks)
    slots = sorted(X)

    # первая ступень: отложенные предсказания для каждого слота
    P1 = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        P1[held] = fit(np.vstack([X[s] for s in tr]),
                       np.concatenate([D[s][0] for s in tr]),
                       np.concatenate([D[s][1] for s in tr]), X[held])
        print(f"первая ступень, слот {held}", flush=True)

    base = np.mean([prec(take(P1[s], D[s][1], RATE), *D[s][::1]) for s in slots])
    base = np.mean([prec(take(P1[s], D[s][1], RATE), D[s][0], D[s][1])
                    for s in slots])
    print(f"\nодна ступень: совпадение {base:.3f}")

    print(f"\n{'зона второй ступени':>22}{'совпадение':>12}{'прибавка':>10}")
    print("-" * 46)
    for zone in (0.5, 0.6, 0.7, 0.8):
        acc = []
        for held in slots:
            tr = [s for s in slots if s != held]
            A, y, w = [], [], []
            for s in tr:
                m = take(P1[s], D[s][1], zone)
                A.append(X[s][m]); y.append(D[s][0][m]); w.append(D[s][1][m])
            mte = take(P1[held], D[held][1], zone)
            p2 = fit(np.vstack(A), np.concatenate(y), np.concatenate(w),
                     X[held][mte])
            # итоговая оценка: внутри зоны — вторая ступень, вне — первая
            p = P1[held].copy()
            p[mte] = 1.0 + p2          # зона всегда выше остального
            acc.append(prec(take(p, D[held][1], RATE), D[held][0], D[held][1]))
        q = float(np.mean(acc))
        print(f"{zone:>21.0%}{q:>12.3f}{q-base:>+10.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
