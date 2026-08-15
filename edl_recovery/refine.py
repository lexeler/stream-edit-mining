#!/usr/bin/env python3
"""
Уточняющий проход.

1. Сшиваем фрагменты, у которых сдвиг одинаковый — они разорваны по ошибке
   из-за погрешности окна ведения.
2. В оставшихся дырах ищем не по всему стриму, а только между соседями.
   Пространство поиска сужается в сотни раз, поэтому порог можно опустить:
   ложное срабатывание там почти невозможно.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
FPS = 100.0
TOL_D = 15          # 0.15 с — считаем сдвиг тем же самым
NCC_LOCAL = 0.60    # порог в суженном поиске
W_MIN = 40          # минимальное окно поиска, 0.4 с
MARGIN = 3000       # запас по стриму, 30 с


def stitch(segs):
    """сшиваем соседей с одинаковым сдвигом"""
    out = [dict(segs[0])]
    for s in segs[1:]:
        p = out[-1]
        if abs(s["d"] - p["d"]) <= TOL_D and s["e0"] - p["e1"] <= 300:
            p["e1"] = max(p["e1"], s["e1"])
            p["ncc"] = max(p["ncc"], s["ncc"])
        else:
            out.append(dict(s))
    return out


def main():
    S = np.fromfile(os.path.join(SP, "stream7.env"), dtype=np.float32)
    E = np.fromfile(os.path.join(SP, "edit.env"), dtype=np.float32)
    N, M = len(S), len(E)
    segs = json.load(open(os.path.join(SP, "final7.json")))["segments"]
    print(f"было фрагментов: {len(segs)}")
    segs = stitch(segs)
    print(f"после сшивки:    {len(segs)}")

    Sd = S.astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(Sd)))
    cs2 = np.concatenate(([0.0], np.cumsum(Sd ** 2)))

    def search(q, lo, hi):
        """ищем q в стриме на отрезке [lo,hi]. -> (позиция, ncc)"""
        w = len(q)
        lo = max(0, lo); hi = min(N - w, hi)
        if hi <= lo or w < W_MIN:
            return -1, 0.0
        q = q - q.mean()
        nq = np.linalg.norm(q)
        if nq < 1e-9:
            return -1, 0.0
        q = q / nq
        span = Sd[lo:hi + w]
        L = len(span)
        nfft = 1 << (L + w).bit_length()
        corr = np.fft.irfft(np.fft.rfft(span, nfft) *
                            np.conj(np.fft.rfft(q, nfft)), nfft)[:L - w + 1]
        s1 = cs[lo + w:lo + L + 1] - cs[lo:lo + L - w + 1]
        s2 = cs2[lo + w:lo + L + 1] - cs2[lo:lo + L - w + 1]
        sd = np.sqrt(np.maximum(s2 - s1 * s1 / w, 1e-9))
        ncc = corr / sd
        k = int(np.argmax(ncc))
        return lo + k, float(ncc[k])

    # собираем дыры
    added = []
    prev_end = 0
    bounds = []
    for i, s in enumerate(segs):
        if s["e0"] > prev_end:
            bounds.append((prev_end, s["e0"], i))
        prev_end = s["e1"]
    if prev_end < M:
        bounds.append((prev_end, M, len(segs)))
    print(f"дыр: {len(bounds)}")

    for g0, g1, idx in bounds:
        if g1 - g0 < W_MIN:
            continue
        left = segs[idx - 1] if idx > 0 else None
        right = segs[idx] if idx < len(segs) else None
        lo = (left["e1"] + left["d"]) - MARGIN if left else 0
        hi = (right["e0"] + right["d"]) + MARGIN if right else N
        if hi - lo < 100:
            lo, hi = lo - MARGIN, hi + MARGIN
        w = min(g1 - g0, 400)
        k, ncc = search(E[g0:g0 + w], lo, hi)
        if k < 0 or ncc < NCC_LOCAL:
            continue
        added.append({"e0": g0, "e1": g1, "d": k - g0,
                      "ncc": round(ncc, 4), "src": "refine"})

    print(f"добавлено в дырах: {len(added)}")
    allseg = sorted(segs + added, key=lambda s: s["e0"])
    allseg = stitch(allseg)
    cov = sum(s["e1"] - s["e0"] for s in allseg) / FPS
    dd = [s["d"] for s in allseg]
    viol = sum(1 for a, b in zip(dd, dd[1:]) if b < a - 50)
    print(f"\nитого фрагментов: {len(allseg)}")
    print(f"покрытие: {cov/60:.2f} из {M/FPS/60:.2f} мин ({cov/(M/FPS)*100:.1f}%)")
    print(f"нарушений монотонности: {viol}")
    json.dump({"fps": FPS, "segments": allseg},
              open(os.path.join(SP, "refined7.json"), "w"))


if __name__ == "__main__":
    main()
