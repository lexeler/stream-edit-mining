#!/usr/bin/env python3
"""
Кривая обучения: упираемся в количество данных или в отсутствие сигнала?

Учим на 1, 2, 3, 4, 5 слотах и смотрим на шестом. Для каждого размера
перебираем несколько случайных наборов, чтобы усреднить.

Растёт с каждым слотом -> нужно больше стримов.
Вышло на полку      -> данных хватает, не хватает видео.
"""
import os
import json, collections, itertools, random
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}
REPEAT = 4


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]
    slots = sorted({r["slot"] for r in rows})

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

    rng = random.Random(0)
    print(f"{'слотов в обучении':>18}{'совпало':>10}{'случайно':>10}"
          f"{'выигрыш':>9}{'AUC':>7}{'разброс':>9}")
    print("-" * 64)
    for n in range(1, len(slots)):
        hits, rnds, aucs = [], [], []
        for held in slots:
            pool = [s for s in slots if s != held]
            combos = list(itertools.combinations(pool, n))
            rng.shuffle(combos)
            for c in combos[:REPEAT]:
                tr = np.isin(sl, c)
                te = sl == held
                m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05,
                                      l2_leaf_reg=8, verbose=0, random_seed=0)
                m.fit(X[tr], y[tr])
                p = np.clip(m.predict(X[te]), 0, 1)
                yt, dt = y[te], dur[te]
                budget = (yt * dt).sum()
                o = np.argsort(-p)
                k = int(np.searchsorted(np.cumsum(dt[o]), budget)) + 1
                sel = np.zeros(len(p), bool); sel[o[:k]] = True
                hits.append((np.minimum(sel.astype(float), yt) * dt)[sel].sum() / budget)
                rnds.append(budget / dt.sum())
                yb = (yt > 0.5).astype(int)
                aucs.append(roc_auc_score(yb, p, sample_weight=dt))
        h, r, a = np.mean(hits), np.mean(rnds), np.mean(aucs)
        print(f"{n:>18}{h*100:>9.1f}%{r*100:>9.1f}%{(h-r)*100:>+8.1f}"
              f"{a:>7.3f}{np.std(hits)*100:>8.1f}")


if __name__ == "__main__":
    main()
