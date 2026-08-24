#!/usr/bin/env python3
"""
Кадры для описания видео-языковой моделью.

Прежние кадры (2 к/с, 224x224 с обрезкой по центру) годились для CLIP,
но для описания словами они слишком мелкие и обрезанные — по краям как раз
происходит то, что интересно. Берём кадр раз в SEC секунд, ширина 512,
пропорции сохраняем.

Считается на Маке: видео лежит здесь, гонять 50 ГБ на сервер незачем.
Один проход декодирования на слот, слоты по очереди — машина не греется.
"""
import os, subprocess, sys, time, json

BASE = os.environ.get("SEM_STREAMS", "work/streams")
OUT = os.path.join(os.environ.get("SEM_DATASET", "work/dataset"), "кадры_описание")
SEC = 12.0
W = 512
SLOTS = ["1", "2", "3", "4", "5", "6", "7"]


def one(slot):
    v = f"{BASE}/{slot}/streams/full_video.mp4"
    out = f"{OUT}/s{slot}"
    if os.path.exists(f"{out}/done"):
        print(f"слот {slot}: уже готов", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", v,
         "-vf", f"fps=1/{SEC},scale={W}:-2", "-q:v", "4",
         f"{out}/f%05d.jpg"], capture_output=True, text=True)
    if r.returncode:
        print(f"слот {slot}: ОШИБКА {r.stderr[-300:]}", flush=True)
        return
    n = len([f for f in os.listdir(out) if f.endswith(".jpg")])
    open(f"{out}/done", "w").write(json.dumps({"n": n, "sec": SEC, "w": W}))
    print(f"слот {slot}: {n} кадров за {time.time()-t0:.0f} с", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or SLOTS:
        one(s)
    print("готово")
