#!/usr/bin/env python3
"""
Предсказания для одного слота моделью, которая его НЕ видела.
Лучший набор: 28 признаков + семантика.
"""
import os
import json, sys, collections
import numpy as np
from catboost import CatBoostRegressor

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}
HELD = sys.argv[1] if len(sys.argv) > 1 else "5"


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]

    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idxs in by.items():
        d = sem[s]
        S[idxs] = np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]])

    X = np.hstack([np.array([[r[k] for k in base] for r in rows], np.float32), S])
    y = np.array([r["used_frac"] for r in rows], np.float32)
    dur = np.array([r["end"] - r["start"] for r in rows], np.float32)
    sl = np.array([r["slot"] for r in rows])

    tr, te = sl != HELD, sl == HELD
    print(f"обучаюсь на {tr.sum()} элементах (без слота {HELD}), "
          f"предсказываю {te.sum()}")
    m = CatBoostRegressor(iterations=2500, depth=6, learning_rate=0.04,
                          l2_leaf_reg=8, verbose=0, random_seed=0)
    m.fit(X[tr], y[tr])
    p = np.clip(m.predict(X[te]), 0, 1)

    idx = np.where(te)[0]
    out = [{"start": rows[i]["start"], "end": rows[i]["end"],
            "type": rows[i]["type"], "score": round(float(s), 4),
            "true": float(rows[i]["used_frac"])}
           for i, s in zip(idx, p)]
    budget = (y[te] * dur[te]).sum()
    print(f"монтаж эталона: {budget/60:.1f} мин из {dur[te].sum()/60:.1f}")
    json.dump({"slot": HELD, "budget_sec": float(budget),
               "duration_sec": float(dur[te].sum()), "items": out},
              open(f"{DATA}/pred_slot{HELD}.json", "w"))
    print(f"сохранено: pred_slot{HELD}.json")


if __name__ == "__main__":
    main()
