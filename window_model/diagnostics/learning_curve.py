#!/usr/bin/env python3
"""
Сколько стримов нужно для нужной точности.

Меряем, как растёт качество при 1, 2, 3, 4, 5 обучающих стримах, и по этой
кривой оцениваем, сколько пар «стрим + монтажный лист» нужно, чтобы выйти
на заданное совпадение. Это единственный способ ответить на вопрос
«достижимо ли 70 %» не гаданием, а замером.

Меряем две вещи: AUC и точность при доле 40 % — именно она и есть
«совпадение» в постановке задачи.
"""
import sys, itertools, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene"
RATE = 0.40
SUBSETS = 4          # сколько случайных подмножеств каждого размера


def prec_at(p, Y, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    keep = np.zeros(len(p), bool); keep[o[c <= rate]] = True
    ov = (keep * Y * WT).sum(); pred = (keep * WT).sum()
    return float(ov / pred) if pred else 0.0


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1 else BLOCKS).split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
    rng = np.random.default_rng(0)
    res = collections.defaultdict(list)
    for held in slots:
        rest = [s for s in slots if s != held]
        for k in range(1, len(rest) + 1):
            combos = list(itertools.combinations(rest, k))
            if len(combos) > SUBSETS:
                idx = rng.choice(len(combos), SUBSETS, replace=False)
                combos = [combos[i] for i in idx]
            for tr in combos:
                m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=0,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(np.vstack([X[s] for s in tr]),
                      np.concatenate([D[s][0] for s in tr]),
                      sample_weight=np.concatenate([D[s][1] for s in tr]))
                p = np.clip(m.predict(X[held]), 0, 1)
                Y, WT = D[held]
                res[k].append((roc_auc_score(Y > 0.5, p),
                               prec_at(p, Y, WT, RATE)))
        print(f"слот {held} посчитан", flush=True)

    ks = sorted(res)
    print(f"\n{'стримов':>9}{'AUC':>8}{'точность при 40 %':>20}{'опытов':>9}")
    print("-" * 46)
    A, P = [], []
    for k in ks:
        a = np.mean([x[0] for x in res[k]]); pr = np.mean([x[1] for x in res[k]])
        A.append(a); P.append(pr)
        print(f"{k:>9}{a:>8.3f}{pr:>20.3f}{len(res[k]):>9}")

    # продление: точность против логарифма числа стримов
    ks = np.array(ks, float); P = np.array(P); A = np.array(A)
    for name, V, target in (("точность", P, 0.70), ("AUC", A, 0.80)):
        b, a0 = np.polyfit(np.log(ks), V, 1)
        need = float(np.exp((target - a0) / b)) if b > 0 else float("inf")
        print(f"\n{name}: рост {b:+.3f} на удвоение-в-e числа стримов")
        print(f"  чтобы выйти на {target}: примерно {need:.0f} стримов "
              f"(перекрёстная проверка)")
    print("\nВажно: это оценка по перекрёстной проверке, она завышает.")
    print("На честном тесте цифра ниже, значит стримов нужно ещё больше.")


if __name__ == "__main__":
    main()
