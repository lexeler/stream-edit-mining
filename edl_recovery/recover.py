#!/usr/bin/env python3
"""
Окончательное восстановление монтажного листа ролика из стрима слота 7.

Пороги откалиброваны опытами:
  настоящее совпадение по огибающей  -> 0.95..1.00
  чужое                              -> ниже 0.85
  подтверждение по кадру видео       -> 0.985..1.000 против 0.17 у контроля
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import Matcher

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
FPS = 100.0

W_GLOBAL = 400       # окно глобального поиска, 4 с
W_VER = 100          # окно ведения, 1 с
NCC_GLOBAL = 0.93    # порог принятия якоря
NCC_VER = 0.70       # порог, что соответствие держится
STEP = 100           # шаг ведения, 1 с
SKIP = 25            # шаг поиска при неудаче, 0.25 с
MIN_SEG = 30         # минимальный фрагмент, 0.3 с


def run():
    S = np.fromfile(os.path.join(SP, "stream7.env"), dtype=np.float32)
    E = np.fromfile(os.path.join(SP, "edit.env"), dtype=np.float32)
    m = Matcher(S, E)
    M, N = len(E), len(S)

    segs = []
    t = 0
    anchors = 0
    while t < M - MIN_SEG:
        w = min(W_GLOBAL, M - t)
        if w < MIN_SEG:
            break
        k, ncc = m.global_search(E[t:t + w])
        anchors += 1
        if ncc < NCC_GLOBAL:
            t += SKIP
            continue
        d = k - t

        # назад: вдруг фрагмент начался раньше, чем мы его поймали
        b = t
        while b - STEP >= 0 and b - STEP + d >= 0:
            if m.verify(b - STEP, b - STEP + d, W_VER) >= NCC_VER:
                b -= STEP
            else:
                break
        lo, hi = max(0, b - STEP), b
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if mid + d >= 0 and m.verify(mid, mid + d, W_VER) >= NCC_VER:
                hi = mid
            else:
                lo = mid
        start = hi
        if segs and start < segs[-1]["e1"]:
            start = segs[-1]["e1"]

        # вперёд
        x = t
        while x + STEP + W_VER <= M and x + STEP + d + W_VER <= N:
            if m.verify(x + STEP, x + STEP + d, W_VER) >= NCC_VER:
                x += STEP
            else:
                break
        lo, hi = x, min(x + STEP, M - W_VER)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if m.verify(mid, mid + d, W_VER) >= NCC_VER:
                lo = mid
            else:
                hi = mid
        end = min(lo + W_VER, M)

        if end - start >= MIN_SEG:
            segs.append({"e0": int(start), "e1": int(end), "d": int(d),
                         "ncc": round(float(ncc), 4)})
            t = end
        else:
            t += SKIP

        if len(segs) and len(segs) % 50 == 0:
            print(f"    {len(segs)} фрагментов, {t/FPS/60:.1f} из {M/FPS/60:.1f} мин")

    cov = sum(s["e1"] - s["e0"] for s in segs) / FPS
    dd = [s["d"] for s in segs]
    viol = sum(1 for a, b in zip(dd, dd[1:]) if b < a - 50)
    print(f"\n  якорных поисков: {anchors}")
    print(f"  фрагментов: {len(segs)}")
    print(f"  покрыто: {cov/60:.2f} мин из {M/FPS/60:.2f} ({cov/(M/FPS)*100:.1f}%)")
    print(f"  нарушений монотонности: {viol}")
    json.dump({"fps": FPS, "segments": segs}, open(os.path.join(SP, "final7.json"), "w"))
    return segs


if __name__ == "__main__":
    run()
