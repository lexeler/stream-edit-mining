#!/usr/bin/env python3
"""
Достаёт кадры из всех стримов для видео-признаков.

Один проход декодирования на слот: 2 кадра в секунду, обрезка до 224x224.
Из тех же кадров сразу считаем дешёвые признаки (движение, яркость,
смена плана) — второй проход по 40 ГБ видео был бы расточительством.

Сами кадры уедут на сервер для CLIP.
"""
import json, os, subprocess, sys, time
import numpy as np

BASE = os.environ.get("SEM_STREAMS", "work/streams")
TMP = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "vid")
FPS = 2.0
SIZE = 224
SLOTS = ["1", "2", "3", "4", "5", "6", "7"]


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def extract(slot):
    v = f"{BASE}/{slot}/streams/full_video.mp4"
    out = f"{TMP}/s{slot}"
    if os.path.exists(f"{out}/done"):
        print(f"слот {slot}: уже готов", flush=True)
        return out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    r = sh(["ffmpeg", "-v", "error", "-i", v,
            "-vf", f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,"
                   f"crop={SIZE}:{SIZE}",
            "-q:v", "6", f"{out}/f%06d.jpg"])
    if r.returncode:
        print(f"слот {slot}: ОШИБКА {r.stderr[-200:]}", flush=True)
        return None
    n = len([f for f in os.listdir(out) if f.endswith(".jpg")])
    sz = sum(os.path.getsize(f"{out}/{f}") for f in os.listdir(out)
             if f.endswith(".jpg")) / 2 ** 20
    print(f"слот {slot}: {n} кадров, {sz:.0f} МБ, {time.time()-t0:.0f} с", flush=True)
    open(f"{out}/done", "w").write(str(n))
    return out


def cheap_features(slot, folder):
    """движение, яркость, смена плана — из уже извлечённых кадров"""
    import struct
    p = f"{TMP}/vfeat_s{slot}.npz"
    if os.path.exists(p):
        return
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    n = len(files)
    # читаем уменьшенные копии через ffmpeg одним потоком: 32x32 серые
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "image2", "-i", f"{folder}/f%06d.jpg",
         "-vf", "scale=32:32,format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8).astype(np.float32)
    k = 32 * 32
    m = len(a) // k
    fr = a[:m * k].reshape(m, k)
    bright = fr.mean(axis=1) / 255.0
    diff = np.zeros(m, dtype=np.float32)
    diff[1:] = np.abs(np.diff(fr, axis=0)).mean(axis=1) / 255.0
    # смена плана: движение сильно выше локального фона
    w = 30
    pad = np.pad(diff, (w, w), mode="edge")
    loc = np.convolve(pad, np.ones(2 * w) / (2 * w), mode="valid")[:m]
    scene = (diff / (loc + 1e-6)).clip(0, 10)
    np.savez(p, bright=bright, motion=diff, scene=scene, n=m)
    print(f"  дешёвые признаки: {m} кадров", flush=True)


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    for s in sys.argv[1:] or SLOTS:
        f = extract(s)
        if f:
            cheap_features(s, f)
    print("\nготово")
