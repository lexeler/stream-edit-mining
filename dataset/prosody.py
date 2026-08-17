#!/usr/bin/env python3
"""
Просодия: высота голоса и её динамика.

Чего не хватало. В конвейере есть громкость, спектральные характеристики
и плотность речи — но нет ВЫСОТЫ голоса. А для человека именно она
отличает оживлённый рассказ от бубнежа: когда интересно, голос гуляет
по диапазону, когда скучно — держится на одной ноте.

Считаем по звуку 16 кГц через автокорреляцию, без сторонних библиотек.
На сетке 1 секунда:

  f0        медианная высота голоса
  f0_std    разброс высоты — оживлённость
  f0_range  размах между 10-м и 90-м процентилем
  f0_slope  средняя скорость изменения высоты
  voiced    доля озвученных кадров
  per       сила периодичности — чистый голос против шума
  e_std     разброс громкости
  e_range   размах громкости
  e_jump    резкие скачки громкости (крик, смех, удар)
"""
import os, sys, subprocess, time
import numpy as np

AUD = os.path.join(os.environ.get("SEM_WORK", "work/window"), "aud")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SR = 16000
WIN, HOP = 0.05, 0.05          # окно и шаг анализа, с
FMIN, FMAX = 70.0, 350.0       # диапазон голоса
GRID = 1.0                     # сетка выдачи, с


def read(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path,
                        "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                       capture_output=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def pitch_track(x):
    """высота и периодичность по автокорреляции, кусками чтобы влезть в память"""
    w, h = int(WIN * SR), int(HOP * SR)
    n = (len(x) - w) // h + 1
    lo, hi = int(SR / FMAX), int(SR / FMIN)
    nfft = 1 << (2 * w - 1).bit_length()
    f0 = np.zeros(n, np.float32); per = np.zeros(n, np.float32)
    en = np.zeros(n, np.float32)
    CH = 20000
    win = np.hanning(w).astype(np.float32)
    for a in range(0, n, CH):
        b = min(a + CH, n)
        idx = (np.arange(a, b)[:, None] * h + np.arange(w)[None, :])
        F = x[idx] * win
        en[a:b] = np.sqrt((F ** 2).mean(1) + 1e-12)
        F = F - F.mean(1, keepdims=True)
        S = np.fft.rfft(F, nfft, axis=1)
        ac = np.fft.irfft(np.abs(S) ** 2, nfft, axis=1)[:, :hi + 1]
        z = ac[:, :1] + 1e-9
        seg = ac[:, lo:hi + 1] / z
        k = seg.argmax(1)
        per[a:b] = seg[np.arange(b - a), k]
        f0[a:b] = SR / (lo + k)
    return f0, per, en


def to_grid(f0, per, en):
    g = int(GRID / HOP)
    n = len(f0) // g
    f0 = f0[:n * g].reshape(n, g); per = per[:n * g].reshape(n, g)
    en = en[:n * g].reshape(n, g)
    voiced = per > 0.35
    out = np.zeros((n, 9), np.float32)
    edb = 20 * np.log10(en + 1e-6)
    for i in range(n):
        v = f0[i][voiced[i]]
        if len(v) >= 2:
            lv = np.log2(v)
            out[i, 0] = float(np.median(v))
            out[i, 1] = float(lv.std() * 12)                 # в полутонах
            out[i, 2] = float((np.quantile(lv, .9) - np.quantile(lv, .1)) * 12)
            out[i, 3] = float(np.abs(np.diff(lv)).mean() * 12)
        out[i, 4] = float(voiced[i].mean())
        out[i, 5] = float(per[i].mean())
        out[i, 6] = float(edb[i].std())
        out[i, 7] = float(edb[i].max() - edb[i].min())
        out[i, 8] = float(np.abs(np.diff(edb[i])).max()) if g > 1 else 0.0
    return out


def main():
    slots = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for s in slots:
        p = f"{OUT}/pros_s{s}.npz"
        if os.path.exists(p):
            print(f"слот {s}: уже есть", flush=True); continue
        f = f"{AUD}/s{s}.flac"
        if not os.path.exists(f):
            print(f"слот {s}: звука нет", flush=True); continue
        t0 = time.time()
        x = read(f)
        f0, per, en = pitch_track(x)
        G = to_grid(f0, per, en)
        np.savez(p, feats=G, grid=GRID,
                 names=["f0", "f0_std", "f0_range", "f0_slope", "voiced",
                        "per", "e_std", "e_range", "e_jump"])
        print(f"слот {s}: {len(G)} секунд за {time.time()-t0:.0f} с", flush=True)
    print("готово")


if __name__ == "__main__":
    main()
