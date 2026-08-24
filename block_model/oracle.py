#!/usr/bin/env python3
"""
Потолок разбивки на реплики: что было бы, если бы мы идеально знали ответ.

Лента должна быть СПЛОШНОЙ, иначе паузы выпадают из выпуска и метрика врёт.
Блок = [конец предыдущего блока .. конец реплики]; паузы длиннее LONG идут
отдельным блоком «тишина».

Меряем при разных значениях GAP: точность и полноту оракула (оставить блок,
если монтажёр взял больше половины). Для сетки 10 с эта цифра была 0,827 —
сравниваем с ней.
"""
import json, glob, os, sys
import numpy as np

LONG = 3.0


def blocks_for(items, gap, long_pause=LONG):
    """сплошная лента блоков: {start, end, text, kind}"""
    out = []
    cur = None          # текущая реплика
    seg_start = 0.0     # начало текущего блока (включая паузу перед ним)
    for r in items:
        if r["type"] == "silence":
            d = r["end"] - r["start"]
            if cur is not None and d >= gap:
                out.append({"start": seg_start, "end": r["start"],
                            "text": " ".join(cur), "kind": "speech"})
                cur = None
                seg_start = r["start"]
            if cur is None and d >= long_pause:
                # длинная пауза — отдельный блок
                if r["start"] > seg_start:
                    out.append({"start": seg_start, "end": r["start"],
                                "text": "", "kind": "pause"})
                out.append({"start": r["start"], "end": r["end"],
                            "text": "", "kind": "pause"})
                seg_start = r["end"]
            continue
        if cur is None:
            cur = []
        cur.append(r["word"])
    last = items[-1]["end"]
    if cur is not None:
        out.append({"start": seg_start, "end": last, "text": " ".join(cur),
                    "kind": "speech"})
    elif last > seg_start:
        out.append({"start": seg_start, "end": last, "text": "", "kind": "pause"})
    return out


def label(blocks, items):
    st = np.array([r["start"] for r in items], np.float64)
    en = np.array([r["end"] for r in items], np.float64)
    fr = np.array([r.get("used_frac", 0.0) for r in items], np.float64)
    for b in blocks:
        lo = np.maximum(st, b["start"]); hi = np.minimum(en, b["end"])
        ov = np.maximum(hi - lo, 0.0)
        d = ov.sum()
        b["used_frac"] = float((ov * fr).sum() / d) if d > 0 else 0.0
        b["dur"] = b["end"] - b["start"]


def main():
    files = sorted(glob.glob(os.path.join(os.environ.get("SEM_DATASET", "work/dataset"), "slot*.labeled.json")))
    print(f"{'GAP':>5}{'блоков':>9}{'медиана с':>11}{'точность':>10}"
          f"{'полнота':>9}{'доля':>8}")
    print("-" * 52)
    for gap in (0.7, 1.2, 2.0, 3.0, 5.0):
        agg = []
        for f in files:
            items = json.load(open(f))["items"]
            bl = blocks_for(items, gap)
            label(bl, items)
            dur = np.array([b["dur"] for b in bl])
            fr = np.array([b["used_frac"] for b in bl])
            keep = fr > 0.5
            ov = (dur * fr)[keep].sum()
            pred = dur[keep].sum()
            true = (dur * fr).sum()
            agg.append((len(bl), np.median(dur), ov / pred, ov / true,
                        pred / dur.sum()))
        a = np.array(agg)
        print(f"{gap:>5.1f}{a[:,0].mean():>9.0f}{a[:,1].mean():>11.1f}"
              f"{a[:,2].mean():>10.3f}{a[:,3].mean():>9.3f}{a[:,4].mean():>8.1%}")
    # контроль: сплошная ли лента
    items = json.load(open(files[0]))["items"]
    bl = blocks_for(items, 2.0)
    holes = sum(1 for a, b in zip(bl, bl[1:]) if abs(a["end"] - b["start"]) > 1e-6)
    print(f"\nконтроль сплошности: разрывов {holes}, "
          f"покрытие {sum(b['end']-b['start'] for b in bl)/items[-1]['end']:.4f}")


if __name__ == "__main__":
    main()
