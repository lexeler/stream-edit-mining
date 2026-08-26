#!/usr/bin/env python3
"""
Ступень 1 без языковой модели: отдельная модель, обученная НА ЗОНАХ.

Важное отличие от того, что уже проверено. Агрегат блочных оценок по зоне
(twostage.py) не даёт ничего — 0,619 против базы 0,620. Но это не то же
самое, что модель, обученная предсказывать ЦЕННОСТЬ ЗОНЫ: у неё другая
цель, меньше шума и признаки усреднены по 30 секундам.

Схема целиком:
  ступень 1 — эта модель отбирает зоны с запасом R1 к бюджету;
  ступень 2 — прежняя блочная оценка чистит паузы и огрызки внутри зон.

Мерим AUC на уровне зон (ориентир: агрегат блочной даёт 0,787) и итоговую
точность при доле монтажёра. Слот 2 не трогаем.

  python3 zonemodel.py --W 30
"""
import json, os, argparse
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def auc_w(score, lab, w):
    o = np.argsort(score)
    lab, w = lab[o], w[o]
    pos, neg = (lab * w).sum(), ((1 - lab) * w).sum()
    if pos == 0 or neg == 0:
        return 0.5
    cn = np.cumsum((1 - lab) * w) - (1 - lab) * w / 2
    return float(((lab * w) * cn).sum() / (pos * neg))


def zonize(X, y, dur, start, kind, W):
    """сворачивает ленту блоков в зоны: взвешенное среднее + форма зоны"""
    mid = start + dur / 2
    z = (mid // W).astype(int)
    K = int(z.max()) + 1
    zd = np.bincount(z, weights=dur, minlength=K)
    ok = zd > 1e-6
    def agg(V):
        M = np.zeros((K, V.shape[1]), np.float32)
        for j in range(V.shape[1]):
            M[:, j] = np.bincount(z, weights=dur * V[:, j], minlength=K)
        return M / np.maximum(zd, 1e-9)[:, None]
    ZX = agg(X)
    zy = np.bincount(z, weights=dur * y, minlength=K) / np.maximum(zd, 1e-9)
    # форма зоны: сколько речи, сколько блоков, самая длинная пауза
    sp = np.bincount(z, weights=dur * kind, minlength=K) / np.maximum(zd, 1e-9)
    nb = np.bincount(z, minlength=K).astype(np.float32)
    pmax = np.zeros(K, np.float32)
    for i in range(len(dur)):
        if kind[i] < 0.5 and dur[i] > pmax[z[i]]:
            pmax[z[i]] = dur[i]
    pos = (np.arange(K) + 0.5) * W / max(float(start[-1] + dur[-1]), 1.0)
    extra = np.column_stack([sp, nb, pmax, pos, zd])
    return (np.hstack([ZX, extra]).astype(np.float32), zy.astype(np.float32),
            zd.astype(np.float32), z, K, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", default="clf", choices=["clf", "reg"])
    a = ap.parse_args()

    D = {}
    for s in SLOTS:
        f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
        X, y, dur, start = f["X"], f["y"], f["dur"], f["start"]
        kind = f["kind"] if "kind" in f.files else np.ones(len(y), np.float32)
        d = np.load(f"{RULES}/final_s{s}.npy", allow_pickle=True).item()
        pb = rank01(d["mix"])
        ZX, zy, zd, z, K, ok = zonize(X, y, dur, start, kind, a.W)
        D[s] = dict(ZX=ZX, zy=zy, zd=zd, z=z, K=K, y=y, dur=dur, p=pb,
                    T=float(dur.sum()), R=float((dur * y).sum() / dur.sum()),
                    ok=ok)

    print(f"зоны по {a.W:.0f} с, цель — ценность зоны, {a.mode}\n")
    hdr = f"{'слот':>5}{'AUC агрег':>11}{'AUC модель':>12}{'база':>8}"
    for R1 in (1.2, 1.3, 1.5, 1.8):
        hdr += f"{'x'+str(R1):>8}"
    print(hdr)

    res = []
    for s in SLOTS:
        tr = [t for t in SLOTS if t != s]
        Xtr = np.vstack([D[t]["ZX"][D[t]["ok"]] for t in tr])
        ytr = np.concatenate([D[t]["zy"][D[t]["ok"]] for t in tr])
        wtr = np.concatenate([D[t]["zd"][D[t]["ok"]] for t in tr])
        q = D[s]
        ps = []
        for sd in range(a.seeds):
            if a.mode == "clf":
                m = CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05,
                                       l2_leaf_reg=20, verbose=0, random_seed=sd,
                                       thread_count=5, rsm=0.3, subsample=0.8,
                                       bootstrap_type="Bernoulli")
                m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
                ps.append(m.predict_proba(q["ZX"])[:, 1])
            else:
                m = CatBoostRegressor(iterations=600, depth=5, learning_rate=0.05,
                                      l2_leaf_reg=20, verbose=0, random_seed=sd,
                                      thread_count=5, rsm=0.3, subsample=0.8,
                                      bootstrap_type="Bernoulli")
                m.fit(Xtr, ytr, sample_weight=wtr)
                ps.append(m.predict(q["ZX"]))
        zs = np.mean(ps, 0)

        # ориентир: агрегат блочной оценки
        zagg = (np.bincount(q["z"], weights=q["dur"] * q["p"], minlength=q["K"])
                / np.maximum(q["zd"], 1e-9))
        lab = (q["zy"] > q["R"]).astype(float)
        a1 = auc_w(zagg[q["ok"]], lab[q["ok"]], q["zd"][q["ok"]])
        a2 = auc_w(zs[q["ok"]], lab[q["ok"]], q["zd"][q["ok"]])

        o = np.argsort(-q["p"]); c = np.cumsum(q["dur"][o])
        keep = np.zeros(len(q["p"]), bool); keep[o[c <= q["R"] * q["T"]]] = True
        base = (q["dur"] * q["y"])[keep].sum() / q["dur"][keep].sum()

        row = [a1, a2, base]
        for R1 in (1.2, 1.3, 1.5, 1.8):
            oz = np.argsort(-zs); cz = np.cumsum(q["zd"][oz])
            zk = np.zeros(q["K"], bool); zk[oz[cz <= R1 * q["R"] * q["T"]]] = True
            inside = zk[q["z"]]
            idx = np.flatnonzero(inside)
            oo = idx[np.argsort(-q["p"][idx])]
            cc = np.cumsum(q["dur"][oo])
            k2 = np.zeros(len(q["p"]), bool); k2[oo[cc <= q["R"] * q["T"]]] = True
            row.append((q["dur"] * q["y"])[k2].sum() / q["dur"][k2].sum())
        res.append(row)
        print(f"{s:>5}{row[0]:>11.3f}{row[1]:>12.3f}{row[2]:>8.3f}" +
              "".join(f"{x:>8.3f}" for x in row[3:]))

    A = np.array(res)
    print(f"{'средн':>5}{A[:,0].mean():>11.3f}{A[:,1].mean():>12.3f}"
          f"{A[:,2].mean():>8.3f}" +
          "".join(f"{A[:,j].mean():>8.3f}" for j in range(3, A.shape[1])))


if __name__ == "__main__":
    main()
