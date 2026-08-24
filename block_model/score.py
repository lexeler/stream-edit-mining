#!/usr/bin/env python3
"""
Оценка решений модели по блокам.

Главная цифра — точность: сколько из НАШЕГО выпуска совпало с монтажёром.
Она честна только при сопоставимой длине, поэтому длина печатается рядом
и сравнивается с монтажёрской. Случайная база — доля монтажёра в эфире.

Формат решений: dict {номер блока: оценка 0..3} или {номер: True/False}.
"""
import os
import json
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")


def load(slot):
    d = json.load(open(f"{DATA}/s{slot}.blocks.json"))
    return d["blocks"]


def metrics(blocks, keep):
    """keep — булев массив по блокам"""
    dur = np.array([b["dur"] for b in blocks])
    fr = np.array([b["used_frac"] for b in blocks])
    ov = float((dur * fr)[keep].sum())
    pred = float(dur[keep].sum())
    true = float((dur * fr).sum())
    T = float(dur.sum())
    return {
        "точность": ov / pred if pred else 0.0,
        "полнота": ov / true if true else 0.0,
        "минут": pred / 60,
        "минут у монтажёра": true / 60,
        "доля": pred / T,
        "случайно": true / T,
        "IoU": ov / (pred + true - ov) if pred + true - ov else 0.0,
    }


def fill_pauses(blocks, score, default=0.0):
    """тишине даём оценку соседей: монтажёр берёт паузы вместе с речью"""
    n = len(blocks)
    out = np.full(n, np.nan)
    for i, b in enumerate(blocks):
        if i in score:
            out[i] = score[i]
    for i, b in enumerate(blocks):
        if np.isnan(out[i]):
            lo = hi = np.nan
            for j in range(i - 1, -1, -1):
                if not np.isnan(out[j]):
                    lo = out[j]; break
            for j in range(i + 1, n):
                if not np.isnan(out[j]):
                    hi = out[j]; break
            v = np.nanmean([x for x in (lo, hi) if not np.isnan(x)]) \
                if not (np.isnan(lo) and np.isnan(hi)) else default
            out[i] = v
    return out


def report(slot, score, title="", thresholds=(0.5, 1.5, 2.5)):
    blocks = load(slot)
    s = fill_pauses(blocks, score)
    dur = np.array([b["dur"] for b in blocks])
    fr = np.array([b["used_frac"] for b in blocks])
    T, true = dur.sum(), (dur * fr).sum()
    print(f"\n=== слот {slot} {title}")
    print(f"эфир {T/60:.0f} мин, монтажёр {true/60:.0f} мин ({true/T:.1%}), "
          f"решений от модели {len(score)} из {len(blocks)} блоков")
    print(f"{'порог':>7}{'минут':>8}{'доля':>8}{'точность':>10}{'полнота':>9}"
          f"{'IoU':>7}{'случайно':>10}")
    best = None
    for t in thresholds:
        keep = s >= t
        if keep.sum() == 0:
            continue
        m = metrics(blocks, keep)
        print(f"{t:>7.1f}{m['минут']:>8.0f}{m['доля']:>8.1%}{m['точность']:>10.3f}"
              f"{m['полнота']:>9.3f}{m['IoU']:>7.3f}{m['случайно']:>10.3f}")
        if best is None or abs(m["минут"] - true / 60) < abs(best[1]["минут"] - true / 60):
            best = (t, m)
    return best


def at_rate(slot, score, rate=None, seed=0):
    """Точность при заданной доле эфира. Если доля не задана — берём долю
    монтажёра, чтобы сравнение с другими способами было честным.

    Оценки грубые, много одинаковых. Границу внутри одной оценки проводим
    случайно с фиксированным зерном и усредняем по нескольким розыгрышам —
    иначе результат зависит от того, в каком порядке лежат блоки.
    """
    blocks = load(slot)
    s = fill_pauses(blocks, score)
    dur = np.array([b["dur"] for b in blocks])
    fr = np.array([b["used_frac"] for b in blocks])
    T = dur.sum()
    if rate is None:
        rate = (dur * fr).sum() / T
    rng = np.random.default_rng(seed)
    accs, recs = [], []
    for _ in range(7):
        key = s + rng.random(len(s)) * 1e-3
        o = np.argsort(-key)
        c = np.cumsum(dur[o]) / T
        keep = np.zeros(len(s), bool)
        keep[o[c <= rate]] = True
        m = metrics(blocks, keep)
        accs.append(m["точность"]); recs.append(m["полнота"])
    return {"точность": float(np.mean(accs)), "полнота": float(np.mean(recs)),
            "доля": rate, "случайно": float((dur * fr).sum() / T)}


def scores_to_segments(blocks, s, thr):
    segs = []
    for b, v in zip(blocks, s):
        if v < thr:
            continue
        if segs and b["start"] - segs[-1][1] < 1e-6:
            segs[-1][1] = b["end"]
        else:
            segs.append([b["start"], b["end"]])
    return segs
