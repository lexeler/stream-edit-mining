#!/usr/bin/env python3
"""
Финальная проверка на слоте 2. Модель его не видела ни разу.
Обучаемся на всех шести обучающих слотах, предсказываем тестовый.
"""
import os
import json, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}


def feats(rows, sem, base):
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idxs in by.items():
        d = sem[s]
        S[idxs] = np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]])
    B = np.array([[r[k] for k in base] for r in rows], dtype=np.float32)
    return np.hstack([B, S])


def main():
    tr_rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    te_rows = [json.loads(l) for l in open(f"{DATA}/test.jsonl")]
    ans = [json.loads(l) for l in open(f"{DATA}/test_answers.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in tr_rows[0] if k not in SKIP]
    assert all(k in te_rows[0] for k in base), "признаки не совпадают"

    Xtr = feats(tr_rows, sem, base)
    Xte = feats(te_rows, sem, base)
    ytr = np.array([r["used_frac"] for r in tr_rows], np.float32)
    yte = np.array([a["used_frac"] for a in ans], np.float32)
    dte = np.array([r["end"] - r["start"] for r in te_rows], np.float32)
    print(f"обучение {Xtr.shape}, тест {Xte.shape}")

    m = CatBoostRegressor(iterations=2500, depth=6, learning_rate=0.04,
                          l2_leaf_reg=8, verbose=0, random_seed=0)
    m.fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, 1)

    budget = (yte * dte).sum()
    o = np.argsort(-p)
    k = int(np.searchsorted(np.cumsum(dte[o]), budget)) + 1
    sel = np.zeros(len(p), bool)
    sel[o[:k]] = True
    hit = (np.minimum(sel.astype(float), yte) * dte)[sel].sum() / budget
    rnd = budget / dte.sum()
    yb = (yte > 0.5).astype(int)
    auc = roc_auc_score(yb, p, sample_weight=dte)

    print(f"\n=== СЛОТ 2, МОДЕЛЬ ЕГО НЕ ВИДЕЛА ===")
    print(f"  стрим {dte.sum()/60:.1f} мин, монтаж {budget/60:.1f} мин "
          f"({rnd*100:.1f}% эфира)")
    print(f"  совпало при равной длине: {hit*100:.1f}%")
    print(f"  случайный тык дал бы:     {rnd*100:.1f}%")
    print(f"  выигрыш: {(hit-rnd)*100:+.1f} п.п.")
    print(f"  AUC: {auc:.3f} (монетка 0.500)")

    json.dump({"slot": "2", "budget_sec": float(budget),
               "duration_sec": float(dte.sum()),
               "hit": float(hit), "random": float(rnd), "auc": float(auc),
               "items": [{"start": r["start"], "end": r["end"],
                          "type": r["type"], "score": round(float(s), 4),
                          "true": float(t)}
                         for r, s, t in zip(te_rows, p, yte)]},
              open(f"{DATA}/pred_slot2.json", "w"))
    print(f"\nсохранено: pred_slot2.json")


if __name__ == "__main__":
    main()
