#!/usr/bin/env python3
"""
Проверка метода на заведомо известном ответе.

Собираем искусственный «монтаж» из самого стрима — режем 10 кусков в известных
местах и склеиваем. Если алгоритм не восстановит эти же куски, значит проблема
в методе, а не в исходных данных.
"""
import numpy as np, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
SR, HOP, WIN, FPS = 8000, 80, 512, 100.0

# (начало в стриме, длительность) в секундах — произвольные, но по возрастанию
TRUTH = [(320, 40), (905, 18), (1740, 62), (3100, 25), (4880, 55),
         (6200, 12), (7450, 70), (9010, 33), (10600, 45), (12300, 28)]


def build():
    x = np.fromfile(os.path.join(SP, "stream.raw"), dtype="<i2")
    parts, pos = [], 0.0
    plan = []
    for s, d in TRUTH:
        a, b = int(s * SR), int((s + d) * SR)
        parts.append(x[a:b])
        plan.append({"edit_start": pos, "stream_start": float(s), "dur": float(d)})
        pos += d
    np.concatenate(parts).tofile(os.path.join(SP, "synth.raw"))
    print(f"  искусственный монтаж: {len(TRUTH)} кусков, {pos:.0f} с")
    return plan


def envelope(raw_path):
    x = np.fromfile(raw_path, dtype="<i2").astype(np.float32)
    n = 1 + (len(x) - WIN) // HOP
    w = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    edges = np.logspace(np.log10(80), np.log10(3800), 25)
    bands = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0] for i in range(24)]
    bands = [b for b in bands if len(b) > 0]
    idx = (np.arange(n) * HOP)[:, None] + np.arange(WIN)[None, :]
    mag = np.abs(np.fft.rfft(x[idx] * w, axis=1))
    be = np.log1p(np.stack([mag[:, b].mean(axis=1) for b in bands], axis=1))
    flux = np.concatenate([[0.0], np.maximum(np.diff(be, axis=0), 0).sum(axis=1)])
    return flux.astype(np.float32)


if __name__ == "__main__":
    plan = build()
    env = envelope(os.path.join(SP, "synth.raw"))
    env.tofile(os.path.join(SP, "synth.env"))

    from match import Matcher
    S = np.fromfile(os.path.join(SP, "stream.env"), dtype=np.float32)
    m = Matcher(S, env)
    print("\n  проверяем середину каждого куска:")
    ok = 0
    for p in plan:
        te = int((p["edit_start"] + p["dur"] / 2) * FPS)
        w = min(400, len(env) - te)
        k, ncc = m.global_search(env[te:te + w])
        want = (p["stream_start"] + p["dur"] / 2)
        got = k / FPS
        good = abs(got - want) < 1.0
        ok += good
        print(f"    ждали стрим {want:8.1f}с | нашли {got:8.1f}с | NCC={ncc:.3f}  "
              f"{'OK' if good else 'МИМО'}")
    print(f"\n  попаданий: {ok}/{len(plan)}")
