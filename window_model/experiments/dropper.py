#!/usr/bin/env python3
"""
Модель, которая выбрасывает мусор, а не отбирает лучшее.

Важная оговорка: простой переворот метки НИЧЕГО не меняет — «вероятность
оставить» и «вероятность выбросить» дают одинаковый порядок кусков.
Поэтому здесь три настоящих отличия, и каждое проверяется контролем:

  1. перекос потерь на нижний конец: модель учится точно узнавать мусор,
     а не точно узнавать хорошее (взвешивание примеров-выбросов);
  2. отбор по классам, а не по доле: цель — уверенность, а не хронометраж;
  3. несколько заходов: выбросили уверенное, переучились на ОСТАТКЕ,
     выбросили ещё. На остатке задача другая, и модель другая.

Контроль: тот же расчёт для простого переворота нынешней модели.
Если специализация не превосходит переворот — значит она бесполезна,
и это надо сказать прямо.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn+icl"
WSEC = 10.0
FRACS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def fit(A, y, w, Xte, mode, seeds=3):
    ps = []
    for sd in range(seeds):
        kw = dict(iterations=600, depth=4, learning_rate=0.05, l2_leaf_reg=20,
                  verbose=0, random_seed=sd, thread_count=5, rsm=0.3,
                  subsample=0.8, bootstrap_type="Bernoulli")
        if mode == "clf":
            m = CatBoostClassifier(auto_class_weights="Balanced", **kw)
            m.fit(A, (y > 0.5).astype(int), sample_weight=w)
            ps.append(m.predict_proba(Xte)[:, 1])
        else:
            m = CatBoostRegressor(**kw)
            m.fit(A, y, sample_weight=w)
            ps.append(np.clip(m.predict(Xte), 0, 1))
    return np.mean(ps, 0)


def removal(order, Y, WT, frac):
    """убираем долю frac по порядку order -> сколько убрано верно"""
    c = np.cumsum(WT[order]) / WT.sum()
    kill = np.zeros(len(Y), bool); kill[order[c <= frac]] = True
    killed = float((kill * WT).sum())
    right = float((kill * (1 - Y) * WT).sum())
    lost = float((kill * Y * WT).sum())
    return right / killed if killed else 0.0, lost / 60, kill


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(WSEC, blocks)
    slots = sorted(X)
    base_drop = np.mean([float(((1 - D[s][0]) * D[s][1]).sum() / D[s][1].sum())
                         for s in slots])

    # --- вариант 1: переворот нынешней модели (контроль)
    Pk = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        Pk[held] = fit(np.vstack([X[s] for s in tr]),
                       np.concatenate([D[s][0] for s in tr]),
                       np.concatenate([D[s][1] for s in tr]), X[held], "reg")
        print(f"контроль, слот {held}", flush=True)

    # --- вариант 2: модель мусора с перекосом веса на выбросы
    Pd = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        y = np.concatenate([1 - D[s][0] for s in tr])
        w = np.concatenate([D[s][1] for s in tr])
        w = w * (1.0 + 2.0 * (y > 0.5))        # мусору втрое больший вес
        Pd[held] = fit(np.vstack([X[s] for s in tr]), y, w, X[held], "reg")
        print(f"модель мусора, слот {held}", flush=True)

    # --- вариант 3: классификация с уравненными классами
    Pc = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        Pc[held] = fit(np.vstack([X[s] for s in tr]),
                       np.concatenate([1 - D[s][0] for s in tr]),
                       np.concatenate([D[s][1] for s in tr]), X[held], "clf")
        print(f"классификация, слот {held}", flush=True)

    print(f"\nмонтажёр выбрасывает {base_drop:.1%} эфира — это и есть "
          f"точность случайного выброса")
    print(f"\n{'выбросить':>11}" + "".join(f"{n:>16}" for n in
          ("переворот", "модель мусора", "классификация")) +
          f"{'потеряно, мин':>15}")
    print("-" * 74)
    for f in FRACS:
        row, lost = [], []
        for P in (Pk, Pd, Pc):
            acc, ls = [], []
            for s in slots:
                Y, WT = D[s]
                o = np.argsort(P[s]) if P is Pk else np.argsort(-P[s])
                a, l, _ = removal(o, Y, WT, f)
                acc.append(a); ls.append(l)
            row.append(np.mean(acc)); lost.append(np.mean(ls))
        print(f"{f:>10.0%}" + "".join(f"{v:>16.3f}" for v in row)
              + f"{lost[1]:>15.1f}")
    sys.stdout.flush()

    # --- вариант 4: в два захода, переучиваясь на остатке
    print("\n=== в два захода: выбросить, переучиться на остатке, выбросить ===")
    print(f"{'первый':>9}{'второй':>9}{'всего убрано':>14}{'верно':>9}"
          f"{'один заход':>13}{'потеряно, мин':>15}")
    print("-" * 70)
    for f1 in (0.15, 0.25):
        for f2 in (0.10, 0.15, 0.20):
            acc2, lost2, acc1 = [], [], []
            for held in slots:
                tr = [s for s in slots if s != held]
                # первый заход — общая модель мусора
                Y, WT = D[held]
                o = np.argsort(-Pd[held])
                _, _, kill1 = removal(o, Y, WT, f1)
                # переучиваемся только на том, что осталось
                A, y, w = [], [], []
                for s in tr:
                    os_ = np.argsort(-Pd[s])
                    _, _, k1 = removal(os_, D[s][0], D[s][1], f1)
                    m = ~k1
                    A.append(X[s][m]); y.append(1 - D[s][0][m])
                    w.append(D[s][1][m])
                rest = ~kill1
                p2 = fit(np.vstack(A), np.concatenate(y), np.concatenate(w),
                         X[held][rest], "reg")
                idx = np.flatnonzero(rest)[np.argsort(-p2)]
                c = np.cumsum(WT[idx]) / WT.sum()
                kill2 = kill1.copy(); kill2[idx[c <= f2]] = True
                killed = float((kill2 * WT).sum())
                right = float((kill2 * (1 - Y) * WT).sum())
                acc2.append(right / killed)
                lost2.append(float((kill2 * Y * WT).sum()) / 60)
                a1, _, _ = removal(np.argsort(-Pd[held]), Y, WT,
                                   killed / WT.sum())
                acc1.append(a1)
            print(f"{f1:>8.0%}{f2:>9.0%}{f1+f2:>13.0%}{np.mean(acc2):>9.3f}"
                  f"{np.mean(acc1):>13.3f}{np.mean(lost2):>15.1f}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
