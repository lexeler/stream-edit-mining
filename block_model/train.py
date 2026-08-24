#!/usr/bin/env python3
"""
Модель на блоках-репликах: форма речи + типы от языковой модели.

Отличие от прежнего конвейера — единица решения. Там была сетка 10 секунд
(потолок точности 0,827), здесь реплика (потолок 0,871), и признаки
считаются на самой реплике, а не усредняются по окну.

Проверка честная: обучаемся на всех слотах, кроме отложенного, и меряем
точность при доле, равной доле монтажёра. Слот 2 не трогаем.

Запускать на сервере:
  python3 train.py --held 7
"""
import json, os, argparse, sys
import numpy as np
from catboost import CatBoostRegressor

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def load(slot):
    z = np.load(f"{DATA}/feat/s{slot}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], z["start"]


def prec_at_rate(y, dur, p, rate, seeds=7):
    """точность при заданной доле эфира, с честным разрывом равенств"""
    rng = np.random.default_rng(0)
    out = []
    T = dur.sum()
    for _ in range(seeds):
        key = p + rng.random(len(p)) * 1e-9
        o = np.argsort(-key)
        c = np.cumsum(dur[o]) / T
        keep = np.zeros(len(p), bool)
        keep[o[c <= rate]] = True
        ov = (dur * y)[keep].sum()
        out.append(ov / dur[keep].sum())
    return float(np.mean(out))


def smooth(g, lam):
    """склейка в крупные куски: Витерби со штрафом за переключение.
    g — выгода оставить блок (может быть отрицательной), lam — штраф.
    Монтажёр берёт сценами: 73 % выпуска лежит в кусках длиннее минуты."""
    n = len(g)
    d0 = np.zeros(n); d1 = np.zeros(n); bk = np.zeros((n, 2), np.int8)
    d0[0], d1[0] = 0.0, g[0]
    for i in range(1, n):
        a, b = d0[i - 1], d1[i - 1] - lam
        d0[i], bk[i, 0] = (b, 1) if b > a else (a, 0)
        a, b = d1[i - 1], d0[i - 1] - lam
        d1[i], bk[i, 1] = (b + g[i], 0) if b > a else (a + g[i], 1)
    st = 1 if d1[-1] > d0[-1] else 0
    keep = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        keep[i] = bool(st); st = bk[i, st]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="7")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()

    tr = [s for s in SLOTS if s != a.held]
    Xtr = np.vstack([load(s)[0] for s in tr])
    ytr = np.concatenate([load(s)[1] for s in tr])
    wtr = np.concatenate([load(s)[2] for s in tr])
    X, y, dur, start = load(a.held)
    rate = float((dur * y).sum() / dur.sum())

    ps = []
    for sd in range(a.seeds):
        m = CatBoostRegressor(iterations=a.iters, depth=a.depth,
                              learning_rate=0.05, l2_leaf_reg=20, verbose=0,
                              random_seed=sd, thread_count=5, rsm=0.3,
                              subsample=0.8, bootstrap_type="Bernoulli")
        m.fit(Xtr, ytr, sample_weight=wtr)
        ps.append(np.clip(m.predict(X), 0, 1))
    p = np.mean(ps, 0)

    print(f"\n=== отложен слот {a.held}: эфир {dur.sum()/60:.0f} мин, "
          f"выпуск монтажёра {(dur*y).sum()/60:.0f} мин ({rate:.1%})")
    print(f"обучение: слоты {'+'.join(tr)}, {Xtr.shape[0]} блоков, "
          f"{Xtr.shape[1]} признаков")
    acc = prec_at_rate(y, dur, p, rate)
    print(f"точность при доле монтажёра: {acc:.3f} (случайно {rate:.3f})")

    print(f"\n{'штраф':>7}{'кусков':>8}{'минут':>8}{'медиана с':>11}"
          f"{'точность':>10}")
    for lam in (0, 2, 5, 15, 40, 100):
        keep = None
        lo, hi = -1.0, 2.0
        for _ in range(26):
            th = (lo + hi) / 2
            keep = smooth((p - th) * dur, lam * dur.mean())
            got = dur[keep].sum() / dur.sum()
            if got > rate:
                lo = th
            else:
                hi = th
            if abs(got - rate) < 0.003:
                break
        ov = (dur * y)[keep].sum()
        lens, i = [], 0
        while i < len(keep):
            if keep[i]:
                j = i
                while j < len(keep) and keep[j]:
                    j += 1
                lens.append(dur[i:j].sum()); i = j
            else:
                i += 1
        lens = np.array(lens) if lens else np.array([0.0])
        print(f"{lam:>7}{len(lens):>8}{dur[keep].sum()/60:>8.0f}"
              f"{np.median(lens):>11.0f}{ov/dur[keep].sum():>10.3f}")

    np.save(f"{DATA}/pred_blocks_s{a.held}.npy",
            {"p": p, "y": y, "dur": dur, "start": start}, allow_pickle=True)
    print(f"\nсохранено: {DATA}/pred_blocks_s{a.held}.npy")


if __name__ == "__main__":
    main()
