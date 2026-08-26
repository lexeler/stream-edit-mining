#!/usr/bin/env python3
"""
Ступень 2: чистка внутри чернового отбора — с учётом РЕШЕНИЯ СОСЕДЕЙ.

Зачем. Самое сильное правило монтажёра (П2): пауза между двумя выброшенными
кусками живёт в 0,6–4,7 % случаев, между двумя оставленными — в 55–74 %.
Нынешняя модель им пользоваться не может: она судит блок по признакам, а не
по тому, что решено вокруг. Двухступенчатость это даёт естественно —
сначала черновое решение, потом чистка, которая его видит.

Замер по twostage.py: при наших зонах идеальная чистка даёт 0,723 против
базы 0,620, то есть в этой ступени лежит +0,10.

Проверяются варианты:
  all   — вторая ступень пересматривает все блоки;
  pause — речь остаётся за первой ступенью, вторая решает только паузы
          (ровно исходная гипотеза: «вторая модель удаляет паузы»);
  плюс второй проход по обновлённому решению.

Черновое решение берётся из честных LOO-оценок (final_s*.npy), поэтому
утечки нет. Слот 2 не трогаем.

  python3 stage2.py --mode all --draft 1.3
"""
import json, os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def draft_keep(p, dur, budget):
    o = np.argsort(-p); c = np.cumsum(dur[o])
    k = np.zeros(len(p), bool); k[o[c <= budget]] = True
    return k


def ctx_feats(keep, p, dur, start, kind, W=(5, 15, 30, 60, 120)):
    """чем окружён блок по ЧЕРНОВОМУ решению (сам блок исключён)"""
    n = len(dur)
    mid = start + dur / 2
    took = keep.astype(np.float64) * dur
    cs_t = np.concatenate([[0.0], np.cumsum(took)])
    cs_d = np.concatenate([[0.0], np.cumsum(dur)])
    cs_p = np.concatenate([[0.0], np.cumsum(p * dur)])
    cols = []
    for w in W:
        lo = np.searchsorted(mid, mid - w)
        hi = np.searchsorted(mid, mid + w)
        tt = cs_t[hi] - cs_t[lo] - took
        dd = cs_d[hi] - cs_d[lo] - dur
        pp = cs_p[hi] - cs_p[lo] - p * dur
        cols.append(np.where(dd > 0, tt / np.maximum(dd, 1e-9), 0.0))
        cols.append(np.where(dd > 0, pp / np.maximum(dd, 1e-9), 0.0))
    # ближайшие соседи и расстояние до взятого
    for lag in (1, 2, 3):
        cols.append(np.concatenate([np.zeros(lag), keep[:-lag]]).astype(float))
        cols.append(np.concatenate([keep[lag:], np.zeros(lag)]).astype(float))
    dl = np.full(n, 1e4); last = -1e4
    for i in range(n):
        if keep[i]:
            last = mid[i]
        dl[i] = mid[i] - last
    dr = np.full(n, 1e4); nxt = 1e4 + mid[-1]
    for i in range(n - 1, -1, -1):
        if keep[i]:
            nxt = mid[i]
        dr[i] = nxt - mid[i]
    cols += [np.minimum(dl, 300), np.minimum(dr, 300),
             np.minimum(dl + dr, 600)]
    # длина непрерывного взятого/выброшенного участка вокруг
    runlen = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and keep[j] == keep[i]:
            j += 1
        runlen[i:j] = dur[i:j].sum()
        i = j
    cols.append(runlen)
    cols += [p, dur, kind.astype(float), rank01(p)]
    return np.column_stack(cols).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "pause"])
    ap.add_argument("--draft", type=float, default=1.3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        B = json.load(open(f"{RULES}/blocks/s{s}.blocks.json"))["blocks"]
        d = np.load(f"{RULES}/final_s{s}.npy", allow_pickle=True).item()
        y, dur, start = d["y"], d["dur"], d["start"]
        kind = np.array([1 if b["kind"] == "speech" else 0 for b in B])
        p = rank01(d["mix"])
        T = float(dur.sum()); R = float((dur * y).sum() / T)
        D[s] = dict(y=y, dur=dur, start=start, kind=kind, p=p, T=T, R=R)

    def acc(q, score):
        k = draft_keep(score, q["dur"], q["R"] * q["T"])
        return float((q["dur"] * q["y"])[k].sum() / q["dur"][k].sum())

    print(f"ступень 2, режим {a.mode}, черновик x{a.draft}\n")
    print(f"{'слот':>5}{'база':>8}" +
          "".join(f"{'проход'+str(i+1):>9}" for i in range(a.passes)))

    cur = {s: D[s]["p"].copy() for s in SLOTS}
    out = {s: [acc(D[s], D[s]["p"])] for s in SLOTS}
    for it in range(a.passes):
        feats, labs, wts = {}, {}, {}
        for s in SLOTS:
            q = D[s]
            k0 = draft_keep(cur[s], q["dur"], a.draft * q["R"] * q["T"])
            feats[s] = ctx_feats(k0, cur[s], q["dur"], q["start"], q["kind"])
            labs[s] = (q["y"] > 0.5).astype(int)
            wts[s] = q["dur"]
        for s in SLOTS:
            tr = [t for t in SLOTS if t != s]
            Xtr = np.vstack([feats[t] for t in tr])
            ytr = np.concatenate([labs[t] for t in tr])
            wtr = np.concatenate([wts[t] for t in tr])
            ps = []
            for sd in range(a.seeds):
                m = CatBoostClassifier(iterations=500, depth=5,
                                       learning_rate=0.05, l2_leaf_reg=20,
                                       verbose=0, random_seed=sd, thread_count=5,
                                       rsm=0.5, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict_proba(feats[s])[:, 1])
            p2 = rank01(np.mean(ps, 0))
            q = D[s]
            if a.mode == "pause":
                # речь остаётся за первой ступенью, вторая решает только паузы
                mix = np.where(q["kind"] == 1, cur[s], p2)
            else:
                mix = p2
            cur[s] = rank01(mix)
            out[s].append(acc(q, cur[s]))

    A = []
    for s in SLOTS:
        print(f"{s:>5}{out[s][0]:>8.3f}" +
              "".join(f"{x:>9.3f}" for x in out[s][1:]))
        A.append(out[s])
    A = np.array(A)
    print(f"{'средн':>5}{A[:,0].mean():>8.3f}" +
          "".join(f"{A[:,j].mean():>9.3f}" for j in range(1, A.shape[1])))


if __name__ == "__main__":
    main()
