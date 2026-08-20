#!/usr/bin/env python3
"""
Та же макро-задача, но с оглядкой на реальный объём данных.

Окон около 4000, но независимых решений монтажёра — примерно 25 макро-кусков
на стрим, то есть 125 на пять обучающих слотов. Бустинг на тысяче признаков
в таком режиме учит шум. Проверяем, не лучше ли:

  --topk K   отбор K признаков по важности ВНУТРИ каждой складки
             (иначе отбор подсмотрит отложенный слот);
  --model ridge   линейная модель с сильной регуляризацией;
  --depth/--iters мельче и короче.

Всё остальное как в windows.py: доля выпуска выбирается самой моделью.
"""
import sys, argparse, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
from windows import get_data, score, take_rate, viterbi_rate, rank01


def fit_predict(Xtr, ytr, wtr, Xte, a, seed=0):
    if a.model == "ridge":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = RidgeCV(alphas=np.logspace(0, 5, 20))
        m.fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
        return np.clip(m.predict((Xte - mu) / sd), 0, 1)
    kw = dict(iterations=a.iters, depth=a.depth, learning_rate=a.lr,
              l2_leaf_reg=a.l2, verbose=0, random_seed=seed, thread_count=5,
              loss_function="RMSE", rsm=a.rsm, subsample=0.8,
              bootstrap_type="Bernoulli")
    m = CatBoostRegressor(**kw)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return np.clip(m.predict(Xte), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=10.0)
    ap.add_argument("--blocks", default="ac+pca")
    ap.add_argument("--model", default="cat")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=20.0)
    ap.add_argument("--rsm", type=float, default=0.3)
    a = ap.parse_args()
    X, D = get_data(a.w, set(a.blocks.split("+")))
    slots = sorted(X)

    P, nsel = {}, []
    for held in slots:
        tr = [s for s in slots if s != held]
        Xtr = np.vstack([X[s] for s in tr])
        ytr = np.concatenate([D[s][0] for s in tr])
        wtr = np.concatenate([D[s][1] for s in tr])
        Xte = X[held]
        if a.topk:
            pre = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.08,
                                    l2_leaf_reg=20, verbose=0, random_seed=0,
                                    thread_count=5, rsm=0.3)
            pre.fit(Xtr, ytr, sample_weight=wtr)
            idx = np.argsort(-pre.get_feature_importance())[:a.topk]
            Xtr, Xte = Xtr[:, idx], Xte[:, idx]
            nsel.append(len(idx))
        P[held] = fit_predict(Xtr, ytr, wtr, Xte, a)

    res = collections.defaultdict(list)
    for held in slots:
        Y, WT = D[held]; p = P[held]
        rate = float((p * WT).sum() / WT.sum())
        res["своя доля"].append(score(take_rate(p, WT, rate), Y, WT))
        res["+Витерби"].append(score(viterbi_rate(rank01(p), WT, 0.2, rate), Y, WT))
        res["_auc"].append(roc_auc_score(Y > 0.5, p))
        res["_rate"].append((rate, float((Y * WT).sum() / WT.sum())))

    rt = np.array(res["_rate"])
    print(f"\n### {a.blocks} | {a.model} | topk={a.topk} d={a.depth} it={a.iters} "
          f"l2={a.l2} rsm={a.rsm} | вход {X[slots[0]].shape[1]}")
    print(f"AUC {np.mean(res['_auc']):.3f} | доля {rt[:,0].mean():.1%} против {rt[:,1].mean():.1%}")
    for k in ("своя доля", "+Витерби"):
        m = np.mean(res[k], 0)
        print(f"{k:>14}  точн {m[0]:.3f}  полн {m[1]:.3f}  F1 {m[2]:.3f}  "
              f"IoU {m[3]:.3f}  оставл {m[4]:.1%}")
    print("  по слотам: " + "  ".join(f"{s}:{v[2]:.3f}"
                                      for s, v in zip(slots, res["своя доля"])))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
