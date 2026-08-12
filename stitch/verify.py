#!/usr/bin/env python3
"""
Проверка склейки: не разъехались ли звук и картинка.

    python3 verify.py <папка>

Метод. Для каждого шва измеряем отдельно, где в результате оказался
момент из исходника — по звуку и по кадрам. Если обе оценки совпали,
звук и видео синхронны.

  источник в момент s виден в результате на (t + La)  по звуку
  источник в момент s виден в результате на (t - Lv)  по кадрам
  рассинхрон = La + Lv          <- знаки противоположны по построению

Обязательно есть контрольный замер вдали от швов. Там обязан быть ноль;
если нет — врёт сам метод, а не склейка.

Предел точности: позицию видео нельзя измерить точнее одного кадра,
то есть +-8.3 мс при 60 fps. Всё, что меньше, считаем нулём.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

SR = 16000


def pcm(p, start, dur):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0,start):.6f}",
                        "-t", f"{dur:.6f}", "-i", p, "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32)


def gray(p, t, w=192, h=108):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.6f}", "-i", p,
                        "-frames:v", "1", "-vf", f"scale={w}:{h},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    return a[:w * h].astype(np.float32) if a.size >= w * h else None


def audio_lag(a, b, max_lag=3.0):
    n = min(len(a), len(b))
    if n < SR:
        return 0.0, 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    N = 1 << (2 * n).bit_length()
    c = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(b, N)), N)
    c /= (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    m = int(max_lag * SR)
    vals = np.concatenate([c[-m:], c[:m]])
    lags = np.concatenate([np.arange(-m, 0), np.arange(0, m)])
    i = int(np.argmax(vals))
    return lags[i] / SR, float(vals[i])


def video_lag(out, src, t_out, t_src, fps, span=6):
    ref = gray(out, t_out)
    if ref is None:
        return None, 0.0
    best, bd = None, 1e18
    for k in range(-span, span + 1):
        g = gray(src, t_src + k / fps)
        if g is None:
            continue
        d = float(np.mean(np.abs(g - ref)))
        if d < bd:
            bd, best = d, k
    return best / fps, bd


def hms(s):
    return f"{int(s//3600)}:{int(s%3600//60):02d}:{s%60:06.3f}"


def verify(folder):
    meta = json.load(open(os.path.join(folder, "offsets.json")))
    out = os.path.join(folder, meta["output"])
    fps = meta["fps"]
    tol = 1.5 / fps
    srcs = {os.path.basename(f): f for f in glob.glob(os.path.join(folder, "*.mp4"))}
    S = meta["sources"]

    print("=== КОНТРОЛЬ (вдали от швов) ===")
    first = srcs[S[0]["file"]]
    ctl_ok = True
    for t in (120.0, 300.0, 500.0):
        if t + 30 > S[0]["dur"]:
            continue
        la, ncc = audio_lag(pcm(out, t, 25.0), pcm(first, t, 25.0))
        lv, dp = video_lag(out, first, t + 1.0, t + 1.0, fps)
        bad = lv is None or abs(la + lv) > tol
        ctl_ok &= not bad
        print(f"  t={t:6.1f}с | звук {la*1000:+7.2f} мс (NCC {ncc:.4f}) | "
              f"видео {lv*1000:+7.2f} мс (пикс {dp:.1f}) | "
              f"рассинхрон {(la+lv)*1000:+6.2f} мс{'  <-- МЕТОД ВРЁТ' if bad else ''}")
    if not ctl_ok:
        print("\n  контроль не прошёл — замерам на швах верить нельзя")

    print("\n=== ШВЫ ===")
    t = 0.0
    ok = True
    for i, s in enumerate(S):
        if i > 0:
            f = srcs[s["file"]]
            la, ncc = audio_lag(pcm(out, t, 25.0), pcm(f, s["skip"], 25.0))
            lv, dp = video_lag(out, f, t + 1.0, s["skip"] + 1.0, fps)
            d = (la + lv) if lv is not None else None
            good = d is not None and abs(d) <= tol and ncc > 0.9
            ok &= good
            print(f"  шов {i} @ {hms(t):>13} | звук {la*1000:+7.2f} мс (NCC {ncc:.4f}) | "
                  f"видео {lv*1000:+7.2f} мс (пикс {dp:.1f}) | "
                  f"рассинхрон {d*1000:+6.2f} мс  {'OK' if good else 'ПРОВЕРЬ'}")
        t += s["dur"] - s["skip"]

    print(f"\nИТОГ: {'швы чистые' if ok and ctl_ok else 'есть на что посмотреть'}")
    print(f"(порог: {tol*1000:.1f} мс = 1.5 кадра; человек замечает рассинхрон примерно от 100 мс)")
    return ok and ctl_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Проверка A/V синхронности склейки")
    ap.add_argument("folder")
    a = ap.parse_args()
    sys.exit(0 if verify(a.folder) else 1)
