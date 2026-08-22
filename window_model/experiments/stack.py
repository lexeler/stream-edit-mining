#!/usr/bin/env python3
"""
Смесь разных по устройству моделей.

Бустинг на рукодельном контексте, свёрточно-рекуррентная сеть по ленте,
дообученный кодировщик текста — все три ошибаются по-разному. Если ошибки
не совпадают, смесь окажется сильнее любой из них.

Предсказания у всех получены честно: каждый слот предсказан моделью,
которая его не видела. Складываем по рангам внутри стрима, потому что
шкалы у моделей разные.
"""
import os, sys, itertools
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov"
WINS = [5.0, 10.0, 20.0, 30.0]
RATE = 0.40


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def to_grid(p, w, n, w0):
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n) + 0.5) * w0
    return np.interp(g, t, p)


def cat_oof(blocks, seeds=3):
    """предсказания бустинга по отложенному слоту, усреднение по окнам"""
    path = f"{DATA}/catoof.npy"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True).item()
        return d["P"], d["D"]
    P, D0 = {}, None
    for w in WINS:
        X, D = get_data(w, set(blocks.split("+")))
        slots = sorted(X)
        for held in slots:
            tr = [s for s in slots if s != held]
            A = np.vstack([X[s] for s in tr])
            y = np.concatenate([D[s][0] for s in tr])
            wt = np.concatenate([D[s][1] for s in tr])
            ps = []
            for sd in range(seeds):
                m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=sd,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(A, y, sample_weight=wt)
                ps.append(np.clip(m.predict(X[held]), 0, 1))
            p = np.mean(ps, 0)
            if w == WINS[0]:
                P.setdefault(held, []).append(p)
            else:
                P[held].append(to_grid(p, w, len(P[held][0]), WINS[0]))
        if w == WINS[0]:
            D0 = D
        print(f"  бустинг, окно {w:g} готово", flush=True)
    P = {s: np.mean(v, 0) for s, v in P.items()}
    np.save(path, {"P": P, "D": D0}, allow_pickle=True)
    return P, D0


def resize(v, n):
    if len(v) == n:
        return v
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


def prec_at(p, Y, WT, rate):
    o = np.argsort(-p); c = np.cumsum(WT[o]) / WT.sum()
    keep = np.zeros(len(p), bool); keep[o[c <= rate]] = True
    ov = (keep * Y * WT).sum(); pred = (keep * WT).sum()
    return float(ov / pred) if pred else 0.0


def main():
    blocks = sys.argv[1] if len(sys.argv) > 1 else BLOCKS
    P, D = cat_oof(blocks)
    slots = sorted(P)
    n = {s: len(P[s]) for s in slots}
    parts = {"бустинг": {s: rank01(P[s]) for s in slots}}
    for name, path in (("сеть", "seqpred.npy"), ("текст", "textft.npy")):
        p = f"{DATA}/{path}"
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True).item()
            if all(s in d for s in slots):
                parts[name] = {s: rank01(resize(np.asarray(d[s], float), n[s]))
                               for s in slots}
    print("в смеси:", ", ".join(parts))

    def measure(w):
        au, pr = [], []
        for s in slots:
            v = sum(w[k] * parts[k][s] for k in parts)
            Y, WT = D[s]
            au.append(roc_auc_score(Y > 0.5, v))
            pr.append(prec_at(v, Y, WT, RATE))
        return np.mean(au), np.mean(pr)

    print(f"\n{'состав':>34}{'AUC':>8}{'точн 40 %':>12}")
    print("-" * 54)
    keys = list(parts)
    best = None
    grid = [0, 0.15, 0.3, 0.5, 0.7, 1.0]
    for combo in itertools.product(grid, repeat=len(keys)):
        if sum(combo) == 0:
            continue
        w = {k: c / sum(combo) for k, c in zip(keys, combo)}
        a, p = measure(w)
        if best is None or p > best[0]:
            best = (p, a, dict(w))
    for k in keys:                       # каждая по отдельности
        a, p = measure({x: (1.0 if x == k else 0.0) for x in keys})
        print(f"{k:>34}{a:>8.3f}{p:>12.3f}")
    a, p = measure({k: 1 / len(keys) for k in keys})
    print(f"{'поровну':>34}{a:>8.3f}{p:>12.3f}")
    p, a, w = best
    nm = " ".join(f"{k}:{v:.2f}" for k, v in w.items() if v > 0)
    print(f"{nm:>34}{a:>8.3f}{p:>12.3f}   <- лучшая")


if __name__ == "__main__":
    main()
