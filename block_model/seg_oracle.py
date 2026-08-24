#!/usr/bin/env python3
"""
Потолок крупноблочного монтажа: сколько даёт идеальный выбор N окон.

Разбор слота 6 показал: 73 % выпуска лежит в 34 кусках длиннее минуты,
внутри взятой сцены остаётся 83 % речи, вне — 2 %. Значит задача может
сводиться не к оценке каждой реплики, а к выбору правильных окон.

Считаем идеальный выбор при разной крупности: максимизируем совпавшее время
при фиксированной доле эфира, со штрафом за каждое включение-выключение.
Штраф задаёт крупность: чем он больше, тем меньше и длиннее куски.

Это верхняя граница для любого способа, который режет крупно.
"""
import json, glob, os
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6"]


def best_keep(frac, dur, lam, mu):
    """максимизируем sum(keep*(frac-lam)*dur) - mu*(число переключений)"""
    n = len(frac)
    g = (frac - lam) * dur
    d0, d1 = 0.0, g[0]
    bk = np.zeros((n, 2), np.int8)
    p0, p1 = np.zeros(n), np.zeros(n)
    p0[0], p1[0] = 0.0, g[0]
    for i in range(1, n):
        a, b = p0[i - 1], p1[i - 1] - mu
        p0[i], bk[i, 0] = (b, 1) if b > a else (a, 0)
        a, b = p1[i - 1], p0[i - 1] - mu
        p1[i], bk[i, 1] = (b + g[i], 0) if b > a else (a + g[i], 1)
    st = 1 if p1[-1] > p0[-1] else 0
    keep = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        keep[i] = bool(st); st = bk[i, st]
    return keep


def solve(frac, dur, rate, mu):
    """подбираем lam так, чтобы доля оставленного была rate"""
    lo, hi = -0.1, 1.1
    keep = None
    for _ in range(30):
        lam = (lo + hi) / 2
        keep = best_keep(frac, dur, lam, mu)
        got = dur[keep].sum() / dur.sum()
        if got > rate:
            lo = lam
        else:
            hi = lam
        if abs(got - rate) < 0.002:
            break
    return keep


def main():
    print(f"{'штраф':>7}{'кусков':>8}{'медиана с':>11}{'точность':>10}"
          f"{'полнота':>9}{'из них >60с':>13}")
    print("-" * 60)
    for mu in (0, 5, 20, 60, 150, 400, 1000):
        rows = []
        for s in SLOTS:
            B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
            dur = np.array([b["dur"] for b in B])
            frac = np.array([b["used_frac"] for b in B])
            rate = (dur * frac).sum() / dur.sum()
            keep = solve(frac, dur, rate, mu)
            ov = (dur * frac)[keep].sum()
            pred = dur[keep].sum()
            # длины непрерывных кусков
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
            rows.append((len(lens), np.median(lens), ov / pred,
                         ov / (dur * frac).sum(),
                         lens[lens > 60].sum() / lens.sum()))
        a = np.array(rows)
        print(f"{mu:>7}{a[:,0].mean():>8.0f}{a[:,1].mean():>11.0f}"
              f"{a[:,2].mean():>10.3f}{a[:,3].mean():>9.3f}{a[:,4].mean():>12.0%}")

    print("\nПоблочный потолок (каждая реплика решается отдельно): 0,871")
    print("У монтажёра кусков: 410–600, но 73 % времени в кусках длиннее минуты.")


if __name__ == "__main__":
    main()
