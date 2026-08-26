#!/usr/bin/env python3
"""
Обучение на парах: какой из двух соседних кусков монтажёр оставил.

Откуда идея. На 761 контрастной границе слепая проверка показала: по разности
формы речи можно угадать, какую сторону выбросили, в 75,8 % случаев. Это
парная задача, и она устроена иначе, чем «предскажи метку куска»: разность
признаков внутри одного стрима не зависит от того, насколько этот эфир
в целом громкий, быстрый или болтливый. Межстримовый сдвиг сокращается.

Учимся на парах блоков из одного стрима, стоящих рядом по времени и решённых
по-разному. На выходе оценка блока — средняя вероятность выиграть у соседей.

  python3 pairwise.py --held 7
"""
import os
import argparse
import numpy as np
from catboost import CatBoostClassifier

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]
NEAR = 180.0     # соседи в пределах трёх минут
MAXP = 40000     # пар на стрим


def load(s):
    z = np.load(f"{DATA}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], z["start"]


def pairs(X, y, start, rng, maxp=MAXP):
    keep = y > 0.5
    drop = y < 0.3
    ki = np.flatnonzero(keep); di = np.flatnonzero(drop)
    if len(ki) == 0 or len(di) == 0:
        return None, None
    A, B = [], []
    for i in ki:
        j = di[np.abs(start[di] - start[i]) <= NEAR]
        if len(j) == 0:
            continue
        if len(j) > 6:
            j = rng.choice(j, 6, replace=False)
        for k in j:
            A.append(i); B.append(k)
    if not A:
        return None, None
    A = np.array(A); B = np.array(B)
    if len(A) > maxp:
        idx = rng.choice(len(A), maxp, replace=False)
        A, B = A[idx], B[idx]
    # половину пар переворачиваем, чтобы не было перекоса по знаку
    flip = rng.random(len(A)) < 0.5
    D = np.where(flip[:, None], X[B] - X[A], X[A] - X[B])
    t = np.where(flip, 0, 1)
    return D.astype(np.float32), t.astype(np.int8)


def score_blocks(model, X, start, rng, n_ref=24):
    """оценка блока — доля побед над случайными соседями по времени"""
    n = len(X)
    out = np.zeros(n)
    order = np.argsort(start)
    for i in range(n):
        lo = np.searchsorted(start[order], start[i] - NEAR)
        hi = np.searchsorted(start[order], start[i] + NEAR)
        cand = order[lo:hi]
        cand = cand[cand != i]
        if len(cand) == 0:
            out[i] = 0.5; continue
        if len(cand) > n_ref:
            cand = rng.choice(cand, n_ref, replace=False)
        D = X[i][None, :] - X[cand]
        out[i] = model.predict_proba(D)[:, 1].mean()
    return out


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0)
    acc = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        keep = np.zeros(len(p), bool)
        keep[o[c <= rate]] = True
        acc.append((dur * y)[keep].sum() / dur[keep].sum())
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="7")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tr = [s for s in SLOTS if s != a.held]
    D, T = [], []
    for s in tr:
        X, y, dur, start = load(s)
        d, t = pairs(X, y, start, rng)
        if d is not None:
            D.append(d); T.append(t)
            print(f"  слот {s}: {len(t)} пар")
    D = np.vstack(D); T = np.concatenate(T)
    print(f"всего пар: {len(T)}, признаков {D.shape[1]}")

    m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.05,
                           l2_leaf_reg=20, verbose=0, random_seed=0,
                           thread_count=5, rsm=0.3, subsample=0.8,
                           bootstrap_type="Bernoulli")
    m.fit(D, T)

    X, y, dur, start = load(a.held)
    rate = float((dur * y).sum() / dur.sum())
    p = score_blocks(m, X, start, rng)
    print(f"\nслот {a.held}: парная модель — точность {prec(y, dur, p, rate):.3f} "
          f"(случайно {rate:.3f})")
    np.save(f"{DATA}/pred_pair_s{a.held}.npy",
            {"p": p, "y": y, "dur": dur, "start": start}, allow_pickle=True)

    # проверим и саму парную задачу: угадывает ли модель, кого выбросили
    d, t = pairs(X, y, start, rng)
    if d is not None:
        acc = (m.predict(d).astype(int) == t).mean()
        print(f"парная точность на отложенном слоте: {acc:.3f} "
              f"(случайно 0,500; на контрастных границах форма речи давала 0,758)")


if __name__ == "__main__":
    main()
