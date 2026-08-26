#!/usr/bin/env python3
"""Типы от языковой модели на равномерной сетке — чтобы их могла увидеть
и прежняя оконная модель, а не только блочная."""
import json, numpy as np, os

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
OUT = DATA + "/typegrid"
TYPES = list("AGRPDWTSFX")
FLAGS = list("!rn")
HOP = 2.0

os.makedirs(OUT, exist_ok=True)
for s in ["1", "2", "3", "4", "5", "6", "7"]:
    B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
    mk = {int(k): v for k, v in
          json.load(open(f"{DATA}/marks/s{s}_m1.json"))["marks"].items()}
    T = B[-1]["end"]
    n = int(np.ceil(T / HOP))
    F = np.zeros((n, len(TYPES) + len(FLAGS)), np.float32)
    last = None
    for i, b in enumerate(B):
        v = mk.get(i) or (last if b["kind"] == "pause" else None)
        if not v:
            continue
        last = v
        i0 = int(b["start"] / HOP); i1 = max(i0 + 1, int(np.ceil(b["end"] / HOP)))
        i1 = min(i1, n)
        F[i0:i1, TYPES.index(v[0])] = 1
        for f in v[1:]:
            if f in FLAGS:
                F[i0:i1, len(TYPES) + FLAGS.index(f)] = 1
    np.savez(f"{OUT}/typ_s{s}.npz", feats=F, hop=HOP,
             names=np.array([f"тип:{t}" for t in TYPES] +
                            [f"пометка:{f}" for f in FLAGS], dtype=object))
    print(f"слот {s}: {n} точек сетки, {F.shape[1]} признаков")
