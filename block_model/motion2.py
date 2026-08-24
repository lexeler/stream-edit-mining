#!/usr/bin/env python3
"""
Движение в кадре подробнее, чем было.

Нынешний признак motion (frames.py) — средняя разница соседних кадров 32x32
в сером. Он дал +0,005 к точности, единственная положительная добавка в
этом заходе. Здесь та же идея, но мер больше, потому что «камера трясётся
при ходьбе», «камера ведёт панораму» и «камера стоит на штативе» — это
разные вещи, а средняя разница их путает.

Считается на кадрах 2 к/с, которые уже лежат в $SEM_WORK/vid/s<N>/:
  mad      — средняя абсолютная разница (как было, но 64x64);
  frac     — доля пикселей, изменившихся заметно (движение объекта против
             общего сдвига картинки);
  p90      — 90-й процентиль разницы: резкое событие в части кадра;
  slow2/6  — разница через 1 и 3 секунды: медленное перемещение;
  still    — кадр почти не изменился (камера стоит);
  bright, dbright — яркость и её перепад (вход-выход, свет).

  python3 motion2.py --slot 7
"""
import os, subprocess, argparse
import numpy as np

VID = os.path.join(os.environ.get("SEM_WORK", "work/window"), "vid")
OUT = os.path.join(os.environ.get("SEM_BLOCKS", "work/blocks"), "motion2")
SZ = 64


def read_gray(folder, n):
    """все кадры одним потоком ffmpeg: SZ x SZ, серые"""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "image2", "-i", f"{folder}/f%06d.jpg",
         "-vf", f"scale={SZ}:{SZ},format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    x = np.frombuffer(r.stdout, np.uint8)
    m = len(x) // (SZ * SZ)
    return x[:m * SZ * SZ].reshape(m, SZ * SZ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    a = ap.parse_args()
    folder = f"{VID}/s{a.slot}"
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    G = read_gray(folder, len(files))
    n = len(G)

    d1 = np.abs(np.diff(G, axis=0))
    mad = np.concatenate([[0.0], d1.mean(1)])
    frac = np.concatenate([[0.0], (d1 > 12).mean(1)])
    p90 = np.concatenate([[0.0], np.percentile(d1, 90, axis=1)])

    def lag(k):
        if n <= k:
            return np.zeros(n, np.float32)
        v = np.abs(G[k:] - G[:-k]).mean(1)
        return np.concatenate([np.zeros(k), v])

    slow2, slow6 = lag(2), lag(6)
    bright = G.mean(1)
    dbright = np.concatenate([[0.0], np.abs(np.diff(bright))])
    thr = np.percentile(mad[mad > 0], 15) if (mad > 0).any() else 0.0
    still = (mad <= thr).astype(np.float32)

    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/s{a.slot}.npz", mad=mad, frac=frac, p90=p90,
             slow2=slow2, slow6=slow6, bright=bright, dbright=dbright,
             still=still, fps=2.0)
    print(f"слот {a.slot}: {n} кадров, 8 мер движения "
          f"(среднее mad {mad.mean():.2f}, стоит {still.mean():.0%} времени)")


if __name__ == "__main__":
    main()
