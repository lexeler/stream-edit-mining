#!/usr/bin/env python3
"""
Первые 20 минут эфира: можно ли выжать там больше.

Замер текущего состояния (кросс-проверка по шести слотам, отбор внутри зоны
с долей монтажёра этой зоны):

    0–10 мин   точность 0,553 при базе 0,391
    10–20 мин  0,700 при 0,573
    0–20 мин   0,675 при 0,482
    весь эфир  0,619 при 0,380

То есть в начале модель работает ХУЖЕ, чем в среднем по эфиру: превышение
над случайным 1,40 против 1,63.

Начало эфира устроено шаблонно (техпроверка -> первое обращение к зрителям ->
приветствие и план -> дорога в ноль -> приезд и плотный монтаж), поэтому
специализированная модель может сработать там, где не сработала для хвоста.

Варианты: общая модель; модель, обученная только на началах; общая плюс
признаки времени от старта; смесь.

  python3 startmodel.py --limit 1200
"""
import os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]
HELD = os.environ.get("HELD", "")


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def prec_in(y, dur, p, seeds=7):
    if dur.sum() <= 0:
        return np.nan
    rate = (dur * y).sum() / dur.sum()
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / max(dur[k].sum(), 1e-9))
    return float(np.mean(out))


def load(s):
    f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
    X, y, dur, start = f["X"], f["y"], f["dur"], f["start"]
    p = f"{RULES}/motion/s{s}.npy"
    if os.path.exists(p):
        X = np.hstack([X, np.load(p, allow_pickle=True).item()["X"]])
    return X.astype(np.float32), y, dur, start


def start_feats(y, dur, start, kind_col):
    """признаки, осмысленные только в начале эфира"""
    mid = start + dur / 2
    sp = kind_col > 0.5
    first_speech = float(mid[sp][0]) if sp.any() else 0.0
    cols = [mid,                                   # секунды от старта
            mid - first_speech,                    # от первой речи эфира
            np.minimum(mid / 60.0, 30),
            np.arange(len(mid), dtype=np.float32), # номер блока от начала
            (mid < 90).astype(np.float32),
            (mid < first_speech).astype(np.float32)]
    return np.column_stack(cols).astype(np.float32)


def fit_predict(Xtr, ytr, wtr, Xte, seeds=5, depth=6, iters=800):
    ps = []
    for sd in range(seeds):
        m = CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.05,
                               l2_leaf_reg=20, verbose=0, random_seed=sd,
                               thread_count=5, rsm=0.3, subsample=0.8,
                               bootstrap_type="Bernoulli")
        m.fit(Xtr, ytr, sample_weight=wtr)
        ps.append(m.predict_proba(Xte)[:, 1])
    return np.mean(ps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=1200.0)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        X, y, dur, start = load(s)
        f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
        kind = f["kind"]
        mid = start + dur / 2
        D[s] = dict(X=X, y=y, dur=dur, start=start, mid=mid,
                    m=(mid < a.limit), S=start_feats(y, dur, start, kind))

    print(f"первые {a.limit/60:.0f} минут эфира, точность внутри зоны\n")
    print(f"{'вариант':<28}" + "".join(f"{'s'+s:>8}" for s in SLOTS) +
          f"{'средн':>9}")

    def show(nm, preds):
        r = [prec_in(D[s]["y"][D[s]["m"]], D[s]["dur"][D[s]["m"]], preds[s])
             for s in SLOTS]
        print(f"{nm:<28}" + "".join(f"{x:>8.3f}" for x in r) +
              f"{np.mean(r):>9.3f}")
        return np.array(r)

    base = {s: (D[s]["dur"][D[s]["m"]] * 0 +
                np.random.default_rng(0).random(D[s]["m"].sum()))
            for s in SLOTS}
    r0 = [((D[s]["dur"] * D[s]["y"])[D[s]["m"]].sum()
           / D[s]["dur"][D[s]["m"]].sum()) for s in SLOTS]
    print(f"{'случайно (база зоны)':<28}" + "".join(f"{x:>8.3f}" for x in r0) +
          f"{np.mean(r0):>9.3f}")

    # 1. общая модель, обученная на всём эфире
    gen = {}
    for s in SLOTS:
        tr = [t for t in SLOTS if t != s]
        Xtr = np.vstack([D[t]["X"] for t in tr])
        ytr = np.concatenate([(D[t]["y"] > 0.5).astype(int) for t in tr])
        wtr = np.concatenate([D[t]["dur"] for t in tr])
        gen[s] = rank01(fit_predict(Xtr, ytr, wtr, D[s]["X"], a.seeds))[D[s]["m"]]
    g = show("общая модель", gen)

    # 2. модель, обученная ТОЛЬКО на началах
    loc = {}
    for s in SLOTS:
        tr = [t for t in SLOTS if t != s]
        Xtr = np.vstack([D[t]["X"][D[t]["m"]] for t in tr])
        ytr = np.concatenate([(D[t]["y"][D[t]["m"]] > 0.5).astype(int) for t in tr])
        wtr = np.concatenate([D[t]["dur"][D[t]["m"]] for t in tr])
        loc[s] = rank01(fit_predict(Xtr, ytr, wtr, D[s]["X"][D[s]["m"]],
                                    a.seeds, depth=5, iters=500))
    l = show("только на началах", loc)

    # 3. только на началах + признаки времени от старта
    loc2 = {}
    for s in SLOTS:
        tr = [t for t in SLOTS if t != s]
        Xtr = np.vstack([np.hstack([D[t]["X"], D[t]["S"]])[D[t]["m"]] for t in tr])
        ytr = np.concatenate([(D[t]["y"][D[t]["m"]] > 0.5).astype(int) for t in tr])
        wtr = np.concatenate([D[t]["dur"][D[t]["m"]] for t in tr])
        Xte = np.hstack([D[s]["X"], D[s]["S"]])[D[s]["m"]]
        loc2[s] = rank01(fit_predict(Xtr, ytr, wtr, Xte, a.seeds,
                                     depth=5, iters=500))
    l2 = show("начала + время от старта", loc2)

    # 4. смеси общей и местной
    for w in (0.3, 0.5, 0.7):
        mix = {s: (1 - w) * rank01(gen[s]) + w * rank01(loc2[s]) for s in SLOTS}
        show(f"смесь: местная {w}", mix)

    # сохраняем оценки, чтобы объединять с судьями-языковыми моделями
    np.save(f"{RULES}/start_preds.npy",
            {s: {"gen": gen[s], "loc": loc[s], "loc2": loc2[s],
                 "y": D[s]["y"][D[s]["m"]], "dur": D[s]["dur"][D[s]["m"]],
                 "mid": D[s]["mid"][D[s]["m"]]} for s in SLOTS},
            allow_pickle=True)
    print(f"\nсохранено: {RULES}/start_preds.npy")


if __name__ == "__main__":
    main()
