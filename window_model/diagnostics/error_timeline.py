#!/usr/bin/env python3
"""
Разбор ошибок: где модель промахивается — россыпью или целыми областями.

Рисуем поминутную ленту: сверху решение монтажёра, снизу решение модели.
Если ошибки идут длинными кусками, значит модель не видит целые сцены,
и лечится это признаками. Если россыпью — лечится сглаживанием.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, score, take_rate

BAR = " ▁▂▃▄▅▆▇█"


def line(v, width):
    """сжимаем вектор до width символов, рисуем плотность"""
    n = len(v)
    idx = (np.arange(width + 1) * n / width).astype(int)
    out = []
    for i in range(width):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        out.append(BAR[int(round(np.clip(v[a:b].mean(), 0, 1) * 8))])
    return "".join(out)


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else
                  "ac+pca+tag+vid+xv+clap").split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
    W = 110
    tot = collections.defaultdict(float)
    for held in slots:
        tr = [s for s in slots if s != held]
        m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                              l2_leaf_reg=20, verbose=0, random_seed=0,
                              thread_count=5, rsm=0.3, subsample=0.8,
                              bootstrap_type="Bernoulli")
        m.fit(np.vstack([X[s] for s in tr]),
              np.concatenate([D[s][0] for s in tr]),
              sample_weight=np.concatenate([D[s][1] for s in tr]))
        p = np.clip(m.predict(X[held]), 0, 1)
        Y, WT = D[held]
        rate = float((p * WT).sum() / WT.sum())
        keep = take_rate(p, WT, rate)
        sc = score(keep, Y, WT)
        mins = WT.sum() / 60
        print(f"\nслот {held}: {mins:.0f} мин, F1 {sc[2]:.3f}, "
              f"оставлено {sc[4]:.0%} против {(Y*WT).sum()/WT.sum():.0%}")
        print("  монтажёр " + line(Y > 0.5, W))
        print("  модель   " + line(keep.astype(float), W))
        err = (keep.astype(int) - (Y > 0.5).astype(int))
        print("  лишнее   " + line((err > 0).astype(float), W))
        print("  упущено  " + line((err < 0).astype(float), W))
        # сколько ошибок приходится на длинные области
        for name, mask in (("лишнее", err > 0), ("упущено", err < 0)):
            runs, k = [], 0
            while k < len(mask):
                if mask[k]:
                    j = k
                    while j < len(mask) and mask[j]:
                        j += 1
                    runs.append(WT[k:j].sum()); k = j
                else:
                    k += 1
            if runs:
                runs = np.array(runs)
                big = runs[runs >= 60].sum() / runs.sum()
                tot[name] += runs.sum()
                print(f"    {name}: {runs.sum()/60:.0f} мин в {len(runs)} кусках, "
                      f"{big:.0%} времени — куски длиннее минуты")
    print(f"\nвсего лишнего {tot['лишнее']/60:.0f} мин, "
          f"упущено {tot['упущено']/60:.0f} мин")


if __name__ == "__main__":
    main()
