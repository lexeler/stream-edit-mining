#!/usr/bin/env python3
"""
Плотная карта соответствия: для каждой точки ролика — откуда она взята.

Склейка = место, где сдвиг скачком меняется. Кандидатов ищем по дешёвой
одномерной огибающей (берём не только лучший, а первые K пиков), решение
принимаем по восьмиполосному признаку — он гораздо разборчивее под музыку.
"""
import json, os, sys
import numpy as np

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
F = 100.0
NB = 8
W = 150          # окно решения, 1.5 с
STEP = 25        # шаг карты, 0.25 с
WB = 40          # окно уточнения границ, 0.4 с
CONT = 0.45      # порог «сдвиг ещё держится»
NEW = 0.65       # порог принятия нового сдвига
TOPK = 40
MARGIN = 0.12   # насколько новый кандидат должен превзойти продолжение        # сколько кандидатов проверяем восьмиполосно


def load():
    E1 = np.fromfile(os.path.join(SP, "edit.env"), dtype=np.float32).astype(np.float64)
    S1 = np.fromfile(os.path.join(SP, "stream7.env"), dtype=np.float32).astype(np.float64)
    E8 = np.fromfile(os.path.join(SP, "edit.b8"), dtype=np.float32).reshape(-1, NB)
    S8 = np.fromfile(os.path.join(SP, "s7.b8"), dtype=np.float32).reshape(-1, NB)
    n = min(len(S1), len(S8)); S1, S8 = S1[:n], S8[:n]
    m = min(len(E1), len(E8)); E1, E8 = E1[:m], E8[:m]
    return E1, S1, E8, S8


class Finder:
    def __init__(self, E1, S1, E8, S8):
        self.E1, self.S1, self.E8, self.S8 = E1, S1, E8, S8
        self.N, self.M = len(S1), len(E1)
        self.cs = np.concatenate(([0.0], np.cumsum(S1)))
        self.cs2 = np.concatenate(([0.0], np.cumsum(S1 ** 2)))
        self.nfft = 1 << (self.N + W).bit_length()
        self.FS = np.fft.rfft(S1, self.nfft)

    def score8(self, te, ts, w=W):
        """сходство восьмиполосных признаков при данном совмещении"""
        if te < 0 or ts < 0 or te + w > self.M or ts + w > self.N:
            return -1.0
        a = self.E8[te:te + w]
        b = self.S8[ts:ts + w]
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        na = np.linalg.norm(a, axis=0)
        nb = np.linalg.norm(b, axis=0)
        ok = (na > 1e-6) & (nb > 1e-6)
        if not ok.any():
            return -1.0
        return float(((a[:, ok] * b[:, ok]).sum(axis=0) / (na[ok] * nb[ok])).mean())

    def candidates(self, te, w=W, k=TOPK):
        """первые k пиков одномерной корреляции"""
        q = self.E1[te:te + w]
        q = q - q.mean()
        nq = np.linalg.norm(q)
        if nq < 1e-9:
            return []
        FQ = np.fft.rfft(q / nq, self.nfft)
        corr = np.fft.irfft(self.FS * np.conj(FQ), self.nfft)[:self.N - w + 1]
        s1 = self.cs[w:] - self.cs[:-w]
        s2 = self.cs2[w:] - self.cs2[:-w]
        sd = np.sqrt(np.maximum(s2 - s1 * s1 / w, 1e-9))
        ncc = corr / sd
        out, tmp = [], ncc.copy()
        for _ in range(k):
            i = int(np.argmax(tmp))
            if tmp[i] < 0.35:
                break
            out.append(i)
            tmp[max(0, i - 150):i + 150] = -9
        return out


def main():
    E1, S1, E8, S8 = load()
    f = Finder(E1, S1, E8, S8)
    M = f.M
    print(f"  ролик {M/F/60:.2f} мин, стрим {f.N/F/60:.2f} мин")

    amap = np.full(M // STEP + 1, -10**9, dtype=np.int64)   # сдвиг для каждой точки
    cur = None
    searches = 0
    for i, t in enumerate(range(0, M - W, STEP)):
        cont = f.score8(t, t + cur) if cur is not None else -1.0
        if cont >= CONT:
            amap[i] = cur
            continue
        best, bs = None, -1.0
        for k in f.candidates(t):
            v = f.score8(t, k)
            if v > bs:
                bs, best = v, k - t
        searches += 1
        # инерция: перепрыгиваем только при заметном выигрыше
        if cur is not None and cont > 0.30 and bs < cont + MARGIN:
            amap[i] = cur
        elif bs >= NEW:
            cur = best
            amap[i] = cur
        else:
            cur = None
        if searches % 200 == 0:
            print(f"    поисков {searches}, дошли до {t/F/60:.1f} мин")

    amap.tofile(os.path.join(SP, "amap.i64"))
    # склеиваем в отрезки по одинаковому сдвигу
    runs = []
    i = 0
    n = len(amap)
    while i < n:
        if amap[i] == -10**9:
            i += 1
            continue
        j = i
        while j + 1 < n and abs(amap[j + 1] - amap[i]) <= 3:
            j += 1
        runs.append({"i0": i, "i1": j, "d": int(amap[i])})
        i = j + 1
    print(f"  участков с постоянным сдвигом: {len(runs)}")

    # уточняем границы: где именно сдвиг переключился
    segs = []
    for r in runs:
        d = r["d"]
        lo = r["i0"] * STEP
        hi = min(r["i1"] * STEP + STEP, M - WB)
        # начало
        a0, a1 = max(0, lo - STEP), lo
        while a1 - a0 > 1:
            mid = (a0 + a1) // 2
            if f.score8(mid, mid + d, WB) >= CONT:
                a1 = mid
            else:
                a0 = mid
        start = a1
        # конец
        b0, b1 = hi, min(hi + STEP, M - WB)
        while b1 - b0 > 1:
            mid = (b0 + b1) // 2
            if f.score8(mid, mid + d, WB) >= CONT:
                b0 = mid
            else:
                b1 = mid
        end = min(b0 + WB, M)
        if segs and start < segs[-1]["e1"]:
            start = segs[-1]["e1"]
        if end - start >= 30:
            segs.append({"e0": int(start), "e1": int(end), "d": int(d)})

    segs.sort(key=lambda s: s["e0"])
    cov = sum(s["e1"] - s["e0"] for s in segs) / F
    print(f"\n  отрезков: {len(segs)}")
    print(f"  покрытие ролика: {cov/60:.2f} из {M/F/60:.2f} мин ({cov/(M/F)*100:.1f}%)")
    json.dump({"fps": F, "segments": segs}, open(os.path.join(SP, "dense7b.json"), "w"))


if __name__ == "__main__":
    main()
