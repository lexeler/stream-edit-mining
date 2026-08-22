#!/usr/bin/env python3
"""
А если монтажёр размечает начало сам?

Главная беда модели — перенос между эфирами: на знакомых стримах 0,58,
на новом 0,51. Разные стримы просто разные. Но в жизни монтажёр всё равно
садится за материал. Что если он размечает первые N минут, а модель
доделывает остальное, уже зная вкус именно этого эфира?

Учим на пяти стримах плюс первые N минут отложенного, меряем на ОСТАВШЕЙСЯ
части. Это честно: размеченный кусок в оценку не входит.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov+surp"
MINUTES = [0, 5, 10, 20, 30, 45]
RATE = 0.40
WSEC = 10.0


def prec_at(p, Y, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    keep = np.zeros(len(p), bool); keep[o[c <= rate]] = True
    ov = (keep * Y * WT).sum(); pred = (keep * WT).sum()
    return float(ov / pred) if pred else 0.0


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(WSEC, blocks)
    slots = sorted(X)
    print(f"{'размечено':>11}{'без разметки':>14}{'с разметкой':>13}"
          f"{'прибавка':>10}")
    print("  сравнение на ОДНОМ И ТОМ ЖЕ остатке стрима")
    print("-" * 48)
    for mins in MINUTES:
        k = int(mins * 60 / WSEC)
        got = {}
        for use in (False, True):
            pr = []
            for held in slots:
                tr = [s for s in slots if s != held]
                A = [X[s] for s in tr]; y = [D[s][0] for s in tr]
                w = [D[s][1] for s in tr]
                if k and use:
                    A.append(X[held][:k]); y.append(D[held][0][:k])
                    w.append(D[held][1][:k] * 3.0)
                ps = []
                for sd in range(3):
                    m = CatBoostRegressor(iterations=600, depth=4,
                                          learning_rate=0.05, l2_leaf_reg=20,
                                          verbose=0, random_seed=sd,
                                          thread_count=5, rsm=0.3, subsample=0.8,
                                          bootstrap_type="Bernoulli")
                    m.fit(np.vstack(A), np.concatenate(y),
                          sample_weight=np.concatenate(w))
                    ps.append(np.clip(m.predict(X[held][k:]), 0, 1))
                p = np.mean(ps, 0)
                Y, WT = D[held][0][k:], D[held][1][k:]
                pr.append(prec_at(p, Y, WT, RATE))
            got[use] = float(np.mean(pr))
            if not k:
                got[True] = got[False]; break
        print(f"{mins:>9} мин{got[False]:>14.3f}{got[True]:>13.3f}"
              f"{got[True]-got[False]:>+10.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
