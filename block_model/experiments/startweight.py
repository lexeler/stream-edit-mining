#!/usr/bin/env python3
"""
Начало эфира: обучение на всём материале, но с повышенным весом начал.

Две крайности уже проверены:
  общая модель (весь эфир, равные веса)     0,679 в первых 20 минутах
  модель только на началах (мало данных)    0,698 на обучающих,
                                            но на слоте 2 провалилась
Здесь середина: учимся на всём, но блокам из первых 20 минут даём вес выше.
Данных остаётся много, а акцент смещается на нужную зону.

  python3 startweight.py
"""
import os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]
LIMIT = 1200.0


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def prec_in(y, dur, p, seeds=7):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    D = {}
    for s in SLOTS:
        X, y, dur, start = load(s)
        D[s] = dict(X=X, y=y, dur=dur, m=(start < LIMIT))

    print(f"вес начал в обучении, точность в первых {LIMIT/60:.0f} минутах\n")
    print(f"{'вес начал':<12}" + "".join(f"{'s'+s:>8}" for s in SLOTS) +
          f"{'средн':>9}")
    for W in (1, 2, 3, 5, 10):
        accs = []
        for s in SLOTS:
            tr = [t for t in SLOTS if t != s]
            Xtr = np.vstack([D[t]["X"] for t in tr])
            ytr = np.concatenate([(D[t]["y"] > 0.5).astype(int) for t in tr])
            wtr = np.concatenate([D[t]["dur"] * np.where(D[t]["m"], W, 1.0)
                                  for t in tr])
            ps = []
            for sd in range(a.seeds):
                m = CatBoostClassifier(iterations=800, depth=6,
                                       learning_rate=0.05, l2_leaf_reg=20,
                                       verbose=0, random_seed=sd, thread_count=5,
                                       rsm=0.3, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict_proba(D[s]["X"])[:, 1])
            p = rank01(np.mean(ps, 0))[D[s]["m"]]
            accs.append(prec_in(D[s]["y"][D[s]["m"]], D[s]["dur"][D[s]["m"]],
                                rank01(p)))
        print(f"{'×'+str(W):<12}" + "".join(f"{x:>8.3f}" for x in accs) +
              f"{np.mean(accs):>9.4f}")


if __name__ == "__main__":
    main()
