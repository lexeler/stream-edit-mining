#!/usr/bin/env python3
"""
Форма монтажа во времени: одинаково ли монтажёр относится к началу, середине
и концу эфира. Если доля оставленного зависит от позиции одинаково у всех
слотов — rel_pos не ловушка, а настоящее правило, и его надо использовать.

Плюс автокорреляция решения: насколько длинными полосами идут решения.
От неё зависит, стоит ли сглаживать предсказания.
"""
import os
import json, collections
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    slots = sorted(by)

    print("=== доля оставленного по децилям эфира ===")
    print(f"{'дециль':>8}" + "".join(f"{s:>8}" for s in slots) + f"{'сред':>8}")
    print("-" * (8 + 8 * len(slots) + 8))
    M = np.zeros((10, len(slots)))
    for j, s in enumerate(slots):
        rr = by[s]
        d = np.array([r["dur"] for r in rr], np.float32)
        f = np.array([r["used_frac"] for r in rr], np.float32)
        t = np.cumsum(d) - d / 2
        dec = np.clip((t / t[-1] * 10).astype(int), 0, 9)
        for k in range(10):
            m = dec == k
            M[k, j] = (d[m] * f[m]).sum() / d[m].sum()
    for k in range(10):
        print(f"{k*10:>5}-{(k+1)*10:<2}" + "".join(f"{M[k,j]:>7.1%}" for j in range(len(slots)))
              + f"{M[k].mean():>8.1%}")

    print("\n=== первые и последние минуты эфира ===")
    print(f"{'минуты':>12}" + "".join(f"{s:>8}" for s in slots) + f"{'сред':>8}")
    for lo, hi, tag in ((0, 1.5, "0-1.5 нач"), (1.5, 5, "1.5-5"), (5, 15, "5-15"),
                        (-15, -5, "конец-15"), (-5, -1.5, "конец-5"), (-1.5, 0, "конец-1.5")):
        vals = []
        for s in slots:
            rr = by[s]
            d = np.array([r["dur"] for r in rr], np.float32)
            f = np.array([r["used_frac"] for r in rr], np.float32)
            t = np.cumsum(d) - d / 2
            T = t[-1]
            a = lo * 60 if lo >= 0 else T + lo * 60
            b = hi * 60 if hi > 0 else T + hi * 60
            m = (t >= a) & (t < b)
            vals.append((d[m] * f[m]).sum() / max(d[m].sum(), 1e-6))
        print(f"{tag:>12}" + "".join(f"{v:>7.1%}" for v in vals) + f"{np.mean(vals):>8.1%}")

    print("\n=== насколько длинными полосами идёт решение ===")
    print("автокорреляция признака 'оставлено' по сетке 10 с")
    print(f"{'сдвиг':>10}{'корреляция':>12}")
    for lag_s in (10, 30, 60, 180, 600, 1800):
        cs = []
        for s in slots:
            rr = by[s]
            d = np.array([r["dur"] for r in rr], np.float32)
            f = np.array([r["used_frac"] for r in rr], np.float32)
            t = np.cumsum(d) - d / 2
            g = np.arange(0, t[-1], 10.0)
            v = np.interp(g, t, f)
            k = lag_s // 10
            if len(v) > k + 10:
                cs.append(np.corrcoef(v[:-k], v[k:])[0, 1])
        print(f"{lag_s:>8} с{np.mean(cs):>12.3f}")


if __name__ == "__main__":
    main()
