#!/usr/bin/env python3
"""
«Обучать на хороших кусках» — проверка гипотезы.

Метка блока — used_frac, доля блока, попавшая в выпуск. Блоки, которых рез
задел частично (0,05 < y < 0,95), — это шум разметки: монтажёр не принимал
по ним отдельного решения, они просто оказались на границе куска. Их около
пятой части ленты, и в обучении они могут только мешать.

Проверяется: обучение без спорных примеров, обучение с их пониженным весом,
обучение только на самых однозначных. Отложенный слот при этом ВСЕГДА
оценивается целиком — иначе сравнение нечестное.

  python3 exp_clean.py
"""
import os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
        D[s] = dict(X=f["X"], y=f["y"], dur=f["dur"])
        D[s]["R"] = float((D[s]["dur"] * D[s]["y"]).sum() / D[s]["dur"].sum())

    variants = {
        "как есть": None,
        "без 0,05-0,95": (0.05, 0.95, 0.0),
        "без 0,2-0,8": (0.2, 0.8, 0.0),
        "спорные весом 0,3": (0.05, 0.95, 0.3),
        "только 0/1": (0.02, 0.98, 0.0),
    }
    print(f"{'вариант':<20}{'доля учебных':>13}" +
          "".join(f"{'s'+s:>8}" for s in SLOTS) + f"{'средн':>9}")
    base = None
    for nm, cfg in variants.items():
        accs, frac = [], []
        for s in SLOTS:
            tr = [t for t in SLOTS if t != s]
            Xs, ys, ws = [], [], []
            for t in tr:
                X, y, dur = D[t]["X"], D[t]["y"], D[t]["dur"]
                w = dur.astype(np.float64).copy()
                if cfg is not None:
                    lo, hi, wt = cfg
                    amb = (y > lo) & (y < hi)
                    if wt == 0.0:
                        X, y, w = X[~amb], y[~amb], w[~amb]
                    else:
                        w[amb] *= wt
                Xs.append(X); ys.append((y > 0.5).astype(int)); ws.append(w)
            Xtr = np.vstack(Xs); ytr = np.concatenate(ys); wtr = np.concatenate(ws)
            frac.append(len(ytr) / sum(len(D[t]["y"]) for t in tr))
            ps = []
            for sd in range(a.seeds):
                m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                                       l2_leaf_reg=20, verbose=0, random_seed=sd,
                                       thread_count=5, rsm=0.3, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict_proba(D[s]["X"])[:, 1])
            accs.append(prec(D[s]["y"], D[s]["dur"], rank01(np.mean(ps, 0)),
                             D[s]["R"]))
        mn = float(np.mean(accs))
        if base is None:
            base = mn
        print(f"{nm:<20}{np.mean(frac):>12.1%} " +
              "".join(f"{x:>8.3f}" for x in accs) + f"{mn:>9.4f}"
              f"{mn-base:>+8.4f}")


if __name__ == "__main__":
    main()
