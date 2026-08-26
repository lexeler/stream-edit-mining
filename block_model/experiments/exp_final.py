#!/usr/bin/env python3
"""
Складываются ли найденные улучшения.

По отдельности (кросс-проверка по шести слотам, база 0,6189):
    движение в кадре                      +0,005
    обучение без спорных примеров 0,2–0,8 +0,004
Складываются ли — вопрос отдельный, здесь он и меряется. Отложенный слот
всегда оценивается целиком, фильтруется только обучающая часть.

  python3 exp_final.py --mdir motion
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
    ap.add_argument("--mdir", default="motion")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
        X, y, dur = f["X"], f["y"], f["dur"]
        p = f"{RULES}/{a.mdir}/s{s}.npy"
        M = np.load(p, allow_pickle=True).item()["X"] if os.path.exists(p) else None
        D[s] = dict(X=X, M=M, y=y, dur=dur,
                    R=float((dur * y).sum() / dur.sum()))

    def build(t, mot):
        return (np.hstack([D[t]["X"], D[t]["M"]]).astype(np.float32)
                if mot and D[t]["M"] is not None else D[t]["X"])

    print(f"движение из каталога {a.mdir}\n")
    print(f"{'вариант':<26}" + "".join(f"{'s'+s:>8}" for s in SLOTS) +
          f"{'средн':>9}{'дельта':>9}")
    base = None
    for nm, mot, clean in (("как есть", False, False),
                           ("движение", True, False),
                           ("фильтр 0,2-0,8", False, True),
                           ("движение + фильтр", True, True)):
        accs = []
        for s in SLOTS:
            tr = [t for t in SLOTS if t != s]
            Xs, ys, ws = [], [], []
            for t in tr:
                X = build(t, mot); y = D[t]["y"]; w = D[t]["dur"].astype(np.float64)
                if clean:
                    keep = ~((y > 0.2) & (y < 0.8))
                    X, y, w = X[keep], y[keep], w[keep]
                Xs.append(X); ys.append((y > 0.5).astype(int)); ws.append(w)
            Xtr = np.vstack(Xs); ytr = np.concatenate(ys); wtr = np.concatenate(ws)
            ps = []
            for sd in range(a.seeds):
                m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                                       l2_leaf_reg=20, verbose=0, random_seed=sd,
                                       thread_count=5, rsm=0.3, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict_proba(build(s, mot))[:, 1])
            accs.append(prec(D[s]["y"], D[s]["dur"], rank01(np.mean(ps, 0)),
                             D[s]["R"]))
        mn = float(np.mean(accs))
        if base is None:
            base = mn
        print(f"{nm:<26}" + "".join(f"{x:>8.3f}" for x in accs) +
              f"{mn:>9.4f}{mn-base:>+9.4f}")


if __name__ == "__main__":
    main()
