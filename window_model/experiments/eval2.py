#!/usr/bin/env python3
"""
Честная оценка. Прошлая была кривая: она хвалила заглушку "взять всё".

Две метрики без этого изъяна.

1. Совпадение при равной длине. Модель обязана ужать эфир ровно до того же
   хронометража, что и монтажёр: берём куски по убыванию оценки, пока не
   наберём нужные минуты. Смотрим, сколько из них монтажёр и правда оставил.
   Случайный тык даст долю, которую монтажёр оставил (~30-47%).

2. AUC — умеет ли модель вообще ранжировать: ставить нужное выше ненужного.
   Не зависит от порога. Монетка = 0.500.
"""
import os
import json, collections
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def at_budget(score, y, d):
    """берём по убыванию оценки, пока не наберём столько же минут"""
    budget = (y * d).sum()
    o = np.argsort(-score)
    cum = np.cumsum(d[o])
    k = int(np.searchsorted(cum, budget)) + 1
    sel = np.zeros(len(y), bool)
    sel[o[:k]] = True
    hit = (np.minimum(sel.astype(float), y) * d)[sel].sum()
    return hit / max(budget, 1e-9)


def main():
    rows = load(f"{DATA}/train.jsonl")
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]
    slots = sorted({r["slot"] for r in rows})

    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idxs in by.items():
        dd = sem[s]
        S[idxs] = np.hstack([dd["near"][dd["idx"]], dd["far"][dd["idx"]]])

    B = np.array([[r[k] for k in base] for r in rows], dtype=np.float32)
    y = np.array([r["used_frac"] for r in rows], dtype=np.float32)
    d = np.array([r["end"] - r["start"] for r in rows], dtype=np.float32)
    sl = np.array([r["slot"] for r in rows])
    ipos = base.index("rel_pos")

    sets = {
        "только 28 признаков": B,
        "28 + семантика": np.hstack([B, S]),
        "семантика без rel_pos": np.hstack([np.delete(B, ipos, 1), S]),
        "только семантика": S,
    }

    print(f"{'вариант':<26}{'слот':>5}{'совпало':>9}{'случайно':>10}{'выигрыш':>9}{'AUC':>7}")
    print("-" * 68)
    agg = collections.defaultdict(list)
    for name, X in sets.items():
        for held in slots:
            tr, te = sl != held, sl == held
            m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05,
                                  l2_leaf_reg=8, verbose=0, random_seed=0)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            hit = at_budget(p, y[te], d[te])
            rnd = (y[te] * d[te]).sum() / d[te].sum()
            yb = (y[te] > 0.5).astype(int)
            auc = roc_auc_score(yb, p, sample_weight=d[te]) if yb.min() != yb.max() else .5
            agg[name].append((hit, rnd, auc))
            print(f"{name if held==slots[0] else '':<26}{held:>5}{hit*100:>8.1f}%"
                  f"{rnd*100:>9.1f}%{(hit-rnd)*100:>+8.1f}{auc:>7.3f}")
        v = agg[name]
        print(f"{'':<26}{'сред':>5}{np.mean([x[0] for x in v])*100:>8.1f}%"
              f"{np.mean([x[1] for x in v])*100:>9.1f}%"
              f"{np.mean([x[0]-x[1] for x in v])*100:>+8.1f}"
              f"{np.mean([x[2] for x in v]):>7.3f}")
        print()


if __name__ == "__main__":
    main()
