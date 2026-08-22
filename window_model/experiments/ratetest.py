#!/usr/bin/env python3
"""
Сколько оставлять: три способа решить это без подсказок.

Разбор ошибок показал, что средняя предсказанная доля — это по сути
константа 36 %, связи с настоящей долей нет (39/26, 29/46, 36/47).
Зато длительность выпусков канала стабильнее самой доли: 75, 71, 56, 66,
104, 64 минуты. Проверяем, не лучше ли ориентироваться на неё.

  средняя доля   rate = среднее предсказание модели (как было)
  хронометраж    rate = типичный выпуск канала / длина этого стрима,
                 типичный берётся ТОЛЬКО с обучающих слотов
  смесь          среднее двух
  оракул         настоящая доля — только чтобы увидеть цену вопроса
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, score, take_rate

BLOCKS = "ac+pca+tag+vid+xv+clap"


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
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

    T = {s: float(D[s][1].sum()) for s in slots}
    true_min = {s: float((D[s][0] * D[s][1]).sum()) / 60 for s in slots}
    print(f"{'слот':>5}{'эфир мин':>10}{'выпуск мин':>12}{'доля':>8}")
    for s in slots:
        print(f"{s:>5}{T[s]/60:>10.0f}{true_min[s]:>12.1f}"
              f"{true_min[s]*60/T[s]:>7.1%}")

    res = collections.defaultdict(list)
    rates = collections.defaultdict(dict)
    for held in slots:
        tr = [s for s in slots if s != held]
        Y, WT = D[held]; p = P[held]
        typ = float(np.median([true_min[s] for s in tr]))       # типичный выпуск
        r_mean = float((p * WT).sum() / WT.sum())
        r_dur = min(typ * 60 / T[held], 0.85)
        r_mix = (r_mean + r_dur) / 2
        r_true = true_min[held] * 60 / T[held]
        for name, r in (("средняя доля", r_mean), ("хронометраж", r_dur),
                        ("смесь", r_mix), ("оракул", r_true)):
            res[name].append(score(take_rate(p, WT, r), Y, WT))
            rates[name][held] = r

    print(f"\n{'способ':>16}{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}"
          f"{'ошибка доли':>13}")
    print("-" * 61)
    for name in ("средняя доля", "хронометраж", "смесь", "оракул"):
        m = np.mean(res[name], 0)
        err = np.mean([abs(rates[name][s] - true_min[s] * 60 / T[s]) for s in slots])
        print(f"{name:>16}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>8.3f}"
              f"{err:>12.1%}")
    print("\nпо слотам, F1:")
    for name in ("средняя доля", "хронометраж", "смесь"):
        print(f"{name:>16}  " + "  ".join(
            f"{s}:{v[2]:.3f}" for s, v in zip(slots, res[name])))
    print("\nпредсказанные доли (хронометраж):  " + "  ".join(
        f"{s}:{rates['хронометраж'][s]:.0%}/{true_min[s]*60/T[s]:.0%}" for s in slots))


if __name__ == "__main__":
    main()
