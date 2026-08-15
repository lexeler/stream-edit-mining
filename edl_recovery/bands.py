#!/usr/bin/env python3
"""Восьмиполосный спектральный признак — намного разборчивее одномерной огибающей."""
import numpy as np, os, sys

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
SR, WIN, HOP = 8000, 512, 80          # 100 кадров/с
NB = 8


def bands(raw_path, out_path):
    x = np.fromfile(raw_path, dtype="<i2").astype(np.float32)
    n = 1 + (len(x) - WIN) // HOP
    w = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    edges = np.logspace(np.log10(100), np.log10(3800), NB + 1)
    sel = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0] for i in range(NB)]

    out = np.zeros((n, NB), dtype=np.float32)
    CH = 20000
    for s in range(0, n, CH):
        c = min(CH, n - s)
        idx = (np.arange(c) * HOP)[:, None] + np.arange(WIN)[None, :] + s * HOP
        mag = np.abs(np.fft.rfft(x[idx] * w, axis=1))
        out[s:s + c] = np.log1p(np.stack([mag[:, b].mean(axis=1) for b in sel], axis=1))

    # убираем медленный дрейф громкости: вычитаем скользящее среднее ~2 с
    k = 200
    pad = np.pad(out, ((k, k), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=0)
    mov = (cs[2 * k:] - cs[:-2 * k]) / (2 * k)
    out = out - mov[:len(out)]
    out.tofile(out_path)
    print(f"  {os.path.basename(out_path)}: {n} кадров x {NB} полос")
    return out


if __name__ == "__main__":
    bands(os.path.join(SP, "edit.raw"), os.path.join(SP, "edit.b8"))
    bands(os.path.join(SP, "s7.raw"), os.path.join(SP, "s7.b8"))
