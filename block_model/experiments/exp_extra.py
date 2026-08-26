#!/usr/bin/env python3
"""
Две добавки к блочной модели, которых в ней ещё не было.

1. ДВИЖЕНИЕ В КАДРЕ (motion/bright/scene). Семантику кадра проверяли —
   описания видео-языковой моделью дают +0,002. Движение это не семантика,
   а форма: идут или стоят. Форма в этой задаче — единственное, что работает.

2. ПРИЗНАКИ УРОВНЯ ЭФИРА. В блочной модели все признаки локальные (окрестность
   до 600 с). Модель не знает, что эфир длинный, а значит резать надо жёстче:
   доля выпуска — функция длины эфира (корреляция −0,83), и режим монтажа
   меняется (эфир 122 мин дырявят, 277 мин режут зонами). Постоянные внутри
   эфира признаки дают дереву возможность калибровать пороги под режим.

  python3 exp_extra.py --which base,motion,glob,both
"""
import json, os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]
TYP_MIN = 68.0        # типичный выпуск канала, минут


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
    ap.add_argument("--which", default="base,motion,glob,both")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
        X, y, dur, start = f["X"], f["y"], f["dur"], f["start"]
        kind = f["kind"]
        T = float(dur.sum())
        M = None
        p = f"{RULES}/motion/s{s}.npy"
        if os.path.exists(p):
            M = np.load(p, allow_pickle=True).item()["X"]
        # признаки уровня эфира — одинаковые для всех блоков слота
        sp = kind > 0.5
        dens = np.array([0.0])
        g = np.array([T / 60.0,
                      float(dur[~sp].sum() / T),
                      float(np.median(dur[sp])) if sp.any() else 0.0,
                      float(len(dur) / (T / 60.0)),
                      float(min(TYP_MIN * 60 / T, 0.85))], np.float32)
        G = np.repeat(g[None, :], len(y), 0)
        D[s] = dict(X=X, y=y, dur=dur, T=T, M=M, G=G,
                    R=float((dur * y).sum() / T))

    print(f"{'вариант':<10}" + "".join(f"{'s'+s:>8}" for s in SLOTS) +
          f"{'средн':>9}{'дельта':>9}")
    base_mean = None
    for which in a.which.split(","):
        accs = []
        for s in SLOTS:
            tr = [t for t in SLOTS if t != s]
            def build(t):
                parts = [D[t]["X"]]
                if which in ("motion", "both") and D[t]["M"] is not None:
                    parts.append(D[t]["M"])
                if which in ("glob", "both"):
                    parts.append(D[t]["G"])
                return np.hstack(parts).astype(np.float32)
            Xtr = np.vstack([build(t) for t in tr])
            ytr = np.concatenate([(D[t]["y"] > 0.5).astype(int) for t in tr])
            wtr = np.concatenate([D[t]["dur"] for t in tr])
            ps = []
            for sd in range(a.seeds):
                m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                                       l2_leaf_reg=20, verbose=0, random_seed=sd,
                                       thread_count=5, rsm=0.3, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict_proba(build(s))[:, 1])
            accs.append(prec(D[s]["y"], D[s]["dur"], rank01(np.mean(ps, 0)),
                             D[s]["R"]))
        mn = float(np.mean(accs))
        if base_mean is None:
            base_mean = mn
        print(f"{which:<10}" + "".join(f"{x:>8.3f}" for x in accs) +
              f"{mn:>9.4f}{mn-base_mean:>+9.4f}")


if __name__ == "__main__":
    main()
