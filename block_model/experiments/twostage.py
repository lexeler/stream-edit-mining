#!/usr/bin/env python3
"""
Двухступенчатая схема (гипотеза):

  ступень 1 — крупные зоны (20/30/60/90 с): какие НЕ брать;
  ступень 2 — внутри взятых зон чистим паузы и огрызки по оценке блока.

Здесь считаются ПОТОЛКИ и базовые варианты, прежде чем тратить языковую
модель. Ключевое число — «идеальный отбор зон + наша нынешняя модель внутри»:
оно говорит, сколько стоит вложение в качество зонного судьи.

Контроли обязательны: случайный отбор зон при той же доле, и чистое
сглаживание оценки по зоне без двухступенчатости.

  ./twostage.py --slots 1,3,4,5,6,7
"""
import json, os, argparse
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def load(slot):
    B = json.load(open(f"{DATA}/s{slot}.blocks.json"))["blocks"]
    d = np.load(f"{DATA}/final_s{slot}.npy", allow_pickle=True).item()
    kind = np.array([1 if b["kind"] == "speech" else 0 for b in B])
    return B, d, kind


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def pick(score, dur, budget, mask=None):
    """жадный отбор по оценке до бюджета времени; mask — что вообще доступно"""
    n = len(score)
    avail = np.ones(n, bool) if mask is None else mask.copy()
    idx = np.flatnonzero(avail)
    o = idx[np.argsort(-score[idx])]
    c = np.cumsum(dur[o])
    keep = np.zeros(n, bool)
    keep[o[c <= budget]] = True
    return keep


def prec(y, dur, keep):
    t = dur[keep].sum()
    return float((dur * y)[keep].sum() / t) if t > 0 else 0.0


def zone_of(mid, W):
    return (mid // W).astype(int)


def two_stage(y, dur, mid, pblock, W, R1, R, T, zscore=None, seed=None):
    """ступень 1: отбор зон; ступень 2: отбор блоков внутри взятых зон"""
    z = zone_of(mid, W)
    K = z.max() + 1
    zdur = np.bincount(z, weights=dur, minlength=K)
    if zscore is None:                       # оценка зоны из оценки блоков
        zs = np.bincount(z, weights=dur * pblock, minlength=K) / np.maximum(zdur, 1e-9)
    elif zscore == "oracle":
        zs = np.bincount(z, weights=dur * y, minlength=K) / np.maximum(zdur, 1e-9)
    elif zscore == "random":
        zs = np.random.default_rng(seed).random(K)
    else:
        zs = zscore
    zkeep = pick(zs, zdur, R1 * T)
    inside = zkeep[z]
    return pick(pblock, dur, R * T, inside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="1,3,4,5,6,7")
    ap.add_argument("--src", default="mix", choices=["mix", "pb", "pw"])
    a = ap.parse_args()
    slots = a.slots.split(",")

    rows = {}
    for s in slots:
        B, d, kind = load(s)
        y, dur, start = d["y"], d["dur"], d["start"]
        mid = start + dur / 2
        T = float(dur.sum())
        R = float((dur * y).sum() / T)          # доля монтажёра — честное сравнение
        p = rank01(d[a.src])
        base = prec(y, dur, pick(p, dur, R * T))
        rows[s] = {"база": base, "случайно": R}

        # 1. потолок: идеальные зоны + идеальные блоки внутри
        for W in (20, 30, 60, 90):
            for R1 in (1.0, 1.3, 1.6, 2.0):
                if R1 * R > 1.0:
                    continue
                k = two_stage(y, dur, mid, y, W, R1 * R, R, T, zscore="oracle")
                rows[s][f"пот W{W} x{R1}"] = prec(y, dur, k)
        # 2. идеальные зоны + НАША модель внутри
        for W in (20, 30, 60, 90):
            for R1 in (1.3, 1.6, 2.0):
                if R1 * R > 1.0:
                    continue
                k = two_stage(y, dur, mid, p, W, R1 * R, R, T, zscore="oracle")
                rows[s][f"орак-зоны W{W} x{R1}"] = prec(y, dur, k)
        # 3. наши зоны (агрегат оценки) + наша модель внутри
        for W in (20, 30, 60, 90):
            for R1 in (1.0, 1.3, 1.6, 2.0):
                if R1 * R > 1.0:
                    continue
                k = two_stage(y, dur, mid, p, W, R1 * R, R, T)
                rows[s][f"свои-зоны W{W} x{R1}"] = prec(y, dur, k)
        # 4. КОНТРОЛЬ: случайные зоны
        for W in (30, 60):
            for R1 in (1.3, 2.0):
                if R1 * R > 1.0:
                    continue
                v = [prec(y, dur, two_stage(y, dur, mid, p, W, R1 * R, R, T,
                                            zscore="random", seed=q))
                     for q in range(5)]
                rows[s][f"СЛУЧ-зоны W{W} x{R1}"] = float(np.mean(v))
        # 5. чистое сглаживание оценки по зоне (без двухступенчатости)
        for W in (20, 30, 60, 90, 120):
            z = zone_of(mid, W)
            K = z.max() + 1
            zdur = np.bincount(z, weights=dur, minlength=K)
            zs = np.bincount(z, weights=dur * p, minlength=K) / np.maximum(zdur, 1e-9)
            for alpha in (0.3, 0.5, 0.7):
                q = alpha * p + (1 - alpha) * zs[z]
                rows[s][f"сглаж W{W} a{alpha}"] = prec(y, dur, pick(q, dur, R * T))

    keys = list(rows[slots[0]].keys())
    print(f"{'вариант':<22}" + "".join(f"{'s'+s:>8}" for s in slots) + f"{'средн':>9}")
    for k in keys:
        v = [rows[s][k] for s in slots]
        star = ""
        if k not in ("база", "случайно"):
            d0 = np.mean(v) - np.mean([rows[s]["база"] for s in slots])
            star = f"  {d0:+.3f}"
        print(f"{k:<22}" + "".join(f"{x:>8.3f}" for x in v) +
              f"{np.mean(v):>9.3f}{star}")


if __name__ == "__main__":
    main()
