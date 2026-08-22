#!/usr/bin/env python3
"""
Обучение с семантикой. Два варианта: с rel_pos и без.

rel_pos дал 46% вклада в модели без семантики и при этом провалил проверку —
он описывает конкретный эфир, а не работу монтажёра. Надо увидеть, нужен ли
он вообще, когда есть смысл текста.

Планка: "не резать вообще" даёт IoU 0.378 / F1 0.544. Не побили — грош цена.
"""
import os
import json, collections
import numpy as np
from catboost import CatBoostRegressor

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def metrics(pred, y, d):
    pred = np.clip(pred, 0, 1)
    inter = (np.minimum(pred, y) * d).sum()
    union = (np.maximum(pred, y) * d).sum()
    kp, kt = (pred * d).sum(), (y * d).sum()
    p = inter / max(kp, 1e-9)
    r = inter / max(kt, 1e-9)
    return {"iou": inter / max(union, 1e-9), "p": p, "r": r,
            "f1": 2 * p * r / max(p + r, 1e-9), "min": kp / 60, "tmin": kt / 60}


def main():
    rows = load(f"{DATA}/train.jsonl")
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]
    slots = sorted({r["slot"] for r in rows})

    # семантика по элементам
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idxs in by.items():
        d = sem[s]
        near, far = d["near"][d["idx"]], d["far"][d["idx"]]
        S[idxs] = np.hstack([near, far])
    print(f"строк {len(rows)}, базовых признаков {len(base)}, семантики {S.shape[1]}")

    B = np.array([[r[k] for k in base] for r in rows], dtype=np.float32)
    y = np.array([r["used_frac"] for r in rows], dtype=np.float32)
    dur = np.array([r["end"] - r["start"] for r in rows], dtype=np.float32)
    sl = np.array([r["slot"] for r in rows])

    ipos = base.index("rel_pos")
    variants = {
        "семантика + все признаки": (np.hstack([B, S]), None),
        "семантика без rel_pos": (np.hstack([np.delete(B, ipos, 1), S]), None),
    }

    print(f"\nпланка (не резать вообще): IoU 0.378  F1 0.544\n")
    agg = collections.defaultdict(list)
    for name, (X, _) in variants.items():
        print(f"--- {name} ---")
        for held in slots:
            tr, te = sl != held, sl == held
            m = CatBoostRegressor(iterations=2000, depth=6, learning_rate=0.04,
                                  l2_leaf_reg=10, verbose=0, random_seed=0)
            m.fit(X[tr], y[tr])
            mm = metrics(m.predict(X[te]), y[te], dur[te])
            agg[name].append(mm)
            print(f"  слот {held}: IoU {mm['iou']:.3f}  P {mm['p']:.3f}  "
                  f"R {mm['r']:.3f}  F1 {mm['f1']:.3f}  "
                  f"минут {mm['min']:6.1f}/{mm['tmin']:6.1f}")
        v = agg[name]
        print(f"  СРЕДНЕЕ: IoU {np.mean([x['iou'] for x in v]):.3f}  "
              f"F1 {np.mean([x['f1'] for x in v]):.3f}\n")

    print("=" * 62)
    for k, v in agg.items():
        i, f = np.mean([x['iou'] for x in v]), np.mean([x['f1'] for x in v])
        print(f"  {k:<28} IoU {i:.3f} ({i-0.378:+.3f})  F1 {f:.3f} ({f-0.544:+.3f})")


if __name__ == "__main__":
    main()
