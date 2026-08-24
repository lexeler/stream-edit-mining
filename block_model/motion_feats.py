#!/usr/bin/env python3
"""
Движение в кадре как признак блока.

Контроль «видео не добавляет» закрывал СЕМАНТИКУ кадра: теги CLIP и описания
видео-языковой моделью (+0,002). Движение — не семантика, а форма: идут или
стоят, трясёт камеру или она замерла. Форма речи в этой задаче работает
(плотность речи 74,6 % на контрастных границах), форму картинки никто
не пробовал — vfeat_s*.npz (bright/motion/scene, 2 к/с) есть в прежнем
конвейере, но в блочные признаки не попадал.

Считает на блок: среднее/макс/разброс, окрестности, отношение к медиане
эфира (важно: у каждого эфира своя камера и свой уровень тряски).

  python3 motion_feats.py --slot 7
"""
import json, os, argparse
import numpy as np

CUT = os.environ.get("SEM_WORK", "work/window")
RULES = os.environ.get("SEM_BLOCKS", "work/blocks")


def roll(vals, mid, dur, sec):
    c = np.concatenate([[0.0], np.cumsum(vals * dur)])
    cd = np.concatenate([[0.0], np.cumsum(dur)])
    lo = np.searchsorted(mid, mid - sec)
    hi = np.searchsorted(mid, mid + sec)
    d = cd[hi] - cd[lo]
    return np.where(d > 0, (c[hi] - c[lo]) / np.maximum(d, 1e-9), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    a = ap.parse_args()
    s = a.slot
    B = json.load(open(f"{RULES}/blocks/s{s}.blocks.json"))["blocks"]
    n = len(B)
    dur = np.array([b["dur"] for b in B], np.float32)
    start = np.array([b["start"] for b in B], np.float32)
    mid = start + dur / 2

    w = np.load(f"{CUT}/vid/vfeat_s{s}.npz")
    keys = [k for k in ("motion", "bright", "scene") if k in w.files]
    FPS = 2.0
    cols, names = [], []
    for k in keys:
        v = w[k].astype(np.float64)
        med = np.median(v[v > 0]) if (v > 0).any() else 1.0
        vn = v / max(med, 1e-6)          # нормировка на эфир: камера у всех разная
        per = np.zeros((n, 3), np.float32)
        for i, b in enumerate(B):
            i0 = int(b["start"] * FPS); i1 = max(i0 + 1, int(b["end"] * FPS))
            i0 = min(i0, len(vn) - 1); i1 = min(i1, len(vn))
            seg = vn[i0:i1]
            per[i] = (seg.mean(), seg.max(), seg.std() if len(seg) > 1 else 0.0)
        cols.append(per)
        names += [f"{k}_mean", f"{k}_max", f"{k}_std"]
        base = per[:, 0]
        for sec in (30, 120, 600):
            r = roll(base, mid, dur, sec).astype(np.float32)
            cols.append(r[:, None]); names.append(f"{k}_ctx{sec}")
        # чем блок отличается от своей окрестности — так же работали lag-признаки
        r120 = roll(base, mid, dur, 120)
        cols.append((base - r120).astype(np.float32)[:, None])
        names.append(f"{k}_dev120")

    X = np.hstack(cols).astype(np.float32)
    os.makedirs(f"{RULES}/motion", exist_ok=True)
    np.save(f"{RULES}/motion/s{s}.npy", {"X": X, "names": names},
            allow_pickle=True)
    print(f"слот {s}: {n} блоков, {X.shape[1]} признаков движения "
          f"({', '.join(keys)})")


if __name__ == "__main__":
    main()
