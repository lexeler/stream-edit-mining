#!/usr/bin/env python3
"""
Контрольный опыт: учимся на одной части стрима, проверяем на другой части
ТОГО ЖЕ стрима.

Он разделяет две совсем разные причины неудачи:
  если внутри стрима точность высокая — признаки содержат нужное, но вкус
    не переносится с эфира на эфир, и лечится это только числом стримов;
  если и внутри низкая — в признаках просто нет того, по чему монтажёр
    решает, и новые стримы не помогут.

Делим стрим на четыре четверти, учимся на трёх, проверяем на четвёртой,
и так по кругу. Рядом печатаем обычную межстримовую проверку.

  python3 insample.py --slot 7
"""
import os
import argparse
import numpy as np
from catboost import CatBoostRegressor

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def load(s):
    z = np.load(f"{DATA}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        keep = np.zeros(len(p), bool)
        keep[o[c <= rate]] = True
        out.append((dur * y)[keep].sum() / dur[keep].sum())
    return float(np.mean(out))


def fit(Xtr, ytr, wtr, Xte, seeds=3):
    ps = []
    for sd in range(seeds):
        m = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05,
                              l2_leaf_reg=20, verbose=0, random_seed=sd,
                              thread_count=5, rsm=0.3, subsample=0.8,
                              bootstrap_type="Bernoulli")
        m.fit(Xtr, ytr, sample_weight=wtr)
        ps.append(np.clip(m.predict(Xte), 0, 1))
    return np.mean(ps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="7")
    a = ap.parse_args()
    X, y, dur = load(a.slot)
    n = len(y)
    rate = float((dur * y).sum() / dur.sum())

    # внутри стрима: четыре куска по времени, каждый по очереди отложен
    accs = []
    edges = [int(n * k / 4) for k in range(5)]
    for k in range(4):
        te = np.zeros(n, bool)
        te[edges[k]:edges[k + 1]] = True
        p = fit(X[~te], y[~te], dur[~te], X[te])
        r = float((dur[te] * y[te]).sum() / dur[te].sum())
        accs.append(prec(y[te], dur[te], p, r))
        print(f"  четверть {k+1}: точность {accs[-1]:.3f} "
              f"(случайно {r:.3f}, блоков {te.sum()})")
    print(f"\nВНУТРИ стрима {a.slot}: {np.mean(accs):.3f}")

    tr = [s for s in SLOTS if s != a.slot]
    Xtr = np.vstack([load(s)[0] for s in tr])
    ytr = np.concatenate([load(s)[1] for s in tr])
    wtr = np.concatenate([load(s)[2] for s in tr])
    p = fit(Xtr, ytr, wtr, X)
    print(f"МЕЖДУ стримами (обучение на {'+'.join(tr)}): "
          f"{prec(y, dur, p, rate):.3f}")
    print(f"случайно: {rate:.3f}")


if __name__ == "__main__":
    main()
