#!/usr/bin/env python3
"""Считает огибающую спектрального потока — признак для сопоставления."""
import numpy as np, os, sys

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
SR = 8000
WIN = 512          # 64 мс
HOP = 80           # 10 мс -> 100 кадров/с
FPS = SR / HOP


def envelope(raw_path, out_path):
    x = np.fromfile(raw_path, dtype="<i2").astype(np.float32)
    n = 1 + (len(x) - WIN) // HOP
    print(f"  {os.path.basename(raw_path)}: {len(x)/SR:.1f} с, {n} кадров")

    w = np.hanning(WIN).astype(np.float32)
    # мел-подобные полосы: 24 логарифмических полосы
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    edges = np.logspace(np.log10(80), np.log10(3800), 25)
    bands = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
             for i in range(24)]
    bands = [b for b in bands if len(b) > 0]

    CH = 20000          # обрабатываем кусками, чтобы влезало в память
    flux = np.zeros(n, dtype=np.float32)
    prev = None
    for start in range(0, n, CH):
        cnt = min(CH, n - start)
        idx = (np.arange(cnt) * HOP)[:, None] + np.arange(WIN)[None, :] + start * HOP
        frames = x[idx] * w
        mag = np.abs(np.fft.rfft(frames, axis=1))
        be = np.stack([mag[:, b].mean(axis=1) for b in bands], axis=1)
        be = np.log1p(be)
        if prev is not None:
            be = np.vstack([prev, be])
        d = np.diff(be, axis=0)
        f = np.maximum(d, 0).sum(axis=1)
        if prev is None:
            f = np.concatenate([[0.0], f])
        flux[start:start + cnt] = f[:cnt]
        prev = be[-1]
        if start % 200000 == 0 and start:
            print(f"    ...{start}/{n}")
    flux.tofile(out_path)
    print(f"  -> {os.path.basename(out_path)} ({n} кадров, {FPS:.0f} к/с)")
    return flux


if __name__ == "__main__":
    for name in ("stream", "edit"):
        envelope(os.path.join(SP, f"{name}.raw"), os.path.join(SP, f"{name}.env"))
