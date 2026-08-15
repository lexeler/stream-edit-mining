#!/usr/bin/env python3
"""
Проверяем каждый фрагмент по картинке и оставляем только подтверждённые.

Звук может обмануть — поверх него в монтаже музыка. Видео независимо:
если кадр из ролика совпадает с кадром стрима, фрагмент настоящий.
Опыт показал разделение 0.985..1.000 у совпадений против 0.17 у случайных пар.
"""
import json, os, subprocess, sys
import numpy as np

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
F = 100.0
THR = 0.85
ST = os.path.join(os.environ.get("SEM_STREAMS", "work/streams"), "7/streams/full_video.mp4")
# опубликованный выпуск, монтажный лист которого восстанавливаем
ED = os.environ.get("SEM_EDITED_VIDEO", "work/sources/edit_youtube_futbolnaya-zaruba.mp4")


def frames(path, t, n=1, w=160, h=90):
    ss = max(0.0, t - (5 / 60 if n > 1 else 0))
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-i", path,
                        "-frames:v", str(n), "-vf", f"scale={w}:{h},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8).astype(np.float32)
    k = w * h
    return [a[i * k:(i + 1) * k] for i in range(len(a) // k)]


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def check(seg):
    """проверяем середину; для длинных — ещё две точки"""
    dur = (seg["e1"] - seg["e0"]) / F
    pts = [0.5] if dur < 8 else [0.25, 0.5, 0.75]
    best = -1.0
    for p in pts:
        te = (seg["e0"] + (seg["e1"] - seg["e0"]) * p) / F
        ts = te + seg["d"] / F
        a = frames(ED, te, 1)
        B = frames(ST, ts, 11)
        if not a or not B:
            continue
        v = max(ncc(a[0], b) for b in B)
        best = max(best, v)
        if best > 0.95:
            break
    return best


def main():
    segs = json.load(open(os.path.join(SP, "refined7.json")))["segments"]
    print(f"проверяю {len(segs)} фрагментов по кадрам...")
    kept, dropped = [], []
    for i, s in enumerate(segs):
        v = check(s)
        s["frame"] = round(v, 4)
        (kept if v >= THR else dropped).append(s)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(segs)}  оставлено {len(kept)}, отсеяно {len(dropped)}")

    cov = sum(s["e1"] - s["e0"] for s in kept) / F
    M = 389929
    dd = [s["d"] for s in kept]
    viol = sum(1 for a, b in zip(dd, dd[1:]) if b < a - 50)
    print(f"\nподтверждено: {len(kept)}   отсеяно: {len(dropped)}")
    print(f"покрытие: {cov/60:.2f} из {M/F/60:.2f} мин ({cov/(M/F)*100:.1f}%)")
    print(f"нарушений монотонности: {viol}")
    src_ref = sum(1 for s in dropped if s.get("src") == "refine")
    print(f"из отсеянных пришло из уточняющего прохода: {src_ref}/{len(dropped)}")
    json.dump({"fps": F, "segments": kept, "dropped": len(dropped)},
              open(os.path.join(SP, "verified7.json"), "w"))


if __name__ == "__main__":
    main()
