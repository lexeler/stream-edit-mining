#!/usr/bin/env python3
"""Ищем, из какого стрима смонтирован ролик. Проверяем все слоты."""
import numpy as np, os, subprocess, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
from selftest import envelope
from match import Matcher

SR, FPS = 8000, 100.0
PROBES = [45, 300, 700, 1200, 1800, 2400, 3000, 3600]   # секунды в ролике


def audio(path, out):
    if os.path.exists(out):
        return
    subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
                    "-ar", str(SR), "-f", "s16le", "-y", out], check=True)


E = np.fromfile(os.path.join(SP, "edit.env"), dtype=np.float32)
print(f"ролик: {len(E)/FPS/60:.1f} мин\n")
print(f"{'слот':>5} {'длина':>9} | " + " ".join(f"{p//60:>2}м" for p in PROBES) + " | вердикт")
print("-" * 78)

results = []
for slot in "1234567":
    v = os.path.join(os.environ.get("SEM_STREAMS", "work/streams"), f"{slot}/streams/full_video.mp4")
    if not os.path.exists(v):
        continue
    raw = os.path.join(SP, f"s{slot}.raw")
    envp = os.path.join(SP, f"s{slot}.env")
    audio(v, raw)
    if os.path.exists(envp):
        S = np.fromfile(envp, dtype=np.float32)
    else:
        S = envelope(raw)
        S.tofile(envp)
    m = Matcher(S, E)
    scores = []
    for p in PROBES:
        t = int(p * FPS)
        w = min(400, len(E) - t)
        if w < 100:
            scores.append(0.0); continue
        k, ncc = m.global_search(E[t:t + w])
        scores.append(ncc)
    hi = sum(1 for s in scores if s > 0.90)
    verdict = "ЭТО ОН" if hi >= len(PROBES) // 2 else ("частично" if hi else "не он")
    print(f"{slot:>5} {len(S)/FPS/60:>7.1f}м | " +
          " ".join(f"{s:.2f}" for s in scores) + f" | {verdict}  (>0.90: {hi})")
    results.append((slot, scores, hi))
    os.remove(raw)

print()
best = max(results, key=lambda r: r[2]) if results else None
if best and best[2] >= 2:
    print(f"источник: слот {best[0]} ({best[2]} проб из {len(PROBES)} выше 0.90)")
else:
    print("ни один слот не подошёл — ролик смонтирован из стрима, которого у нас нет")
