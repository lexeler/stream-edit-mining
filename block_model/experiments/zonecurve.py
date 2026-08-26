#!/usr/bin/env python3
"""
Сколько должен уметь зонный судья, чтобы схема окупилась.

twostage.py показал: идеальные зоны 30 с с запасом 1,3 + нынешняя чистка
внутри дают 0,78 против базы 0,62. Но идеальный судья недостижим. Здесь
строится кривая: качество судьи (AUC на уровне зон) -> итоговая точность.

Судья моделируется как смесь оракула и нашей нынешней оценки, а также как
оракул, испорченный шумом. Обе кривые должны сойтись — это контроль того,
что мерим качество судьи, а не форму подделки.

  ./zonecurve.py
"""
import json, os, argparse
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def auc(score, lab, w=None):
    """взвешенный AUC (лаб. 0/1)"""
    o = np.argsort(score)
    lab = lab[o]
    w = np.ones(len(lab)) if w is None else w[o]
    pos = (lab * w).sum(); neg = ((1 - lab) * w).sum()
    if pos == 0 or neg == 0:
        return 0.5
    cum_neg = np.cumsum((1 - lab) * w) - (1 - lab) * w / 2
    return float(((lab * w) * cum_neg).sum() / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=30.0)
    ap.add_argument("--R1", type=float, default=1.3)
    a = ap.parse_args()

    print(f"зоны по {a.W:.0f} с, запас бюджета x{a.R1}\n")
    print(f"{'судья':<26}{'AUC зон':>9}" +
          "".join(f"{'s'+s:>8}" for s in SLOTS) + f"{'средн':>9}")

    S = {}
    for s in SLOTS:
        B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
        d = np.load(f"{DATA}/final_s{s}.npy", allow_pickle=True).item()
        y, dur, start = d["y"], d["dur"], d["start"]
        mid = start + dur / 2
        T = float(dur.sum()); R = float((dur * y).sum() / T)
        p = rank01(d["mix"])
        z = (mid // a.W).astype(int); K = z.max() + 1
        zdur = np.bincount(z, weights=dur, minlength=K)
        ztrue = np.bincount(z, weights=dur * y, minlength=K) / np.maximum(zdur, 1e-9)
        zours = np.bincount(z, weights=dur * p, minlength=K) / np.maximum(zdur, 1e-9)
        S[s] = dict(y=y, dur=dur, p=p, z=z, K=K, zdur=zdur, ztrue=ztrue,
                    zours=zours, T=T, R=R)

    def run(name, mk):
        accs, aucs = [], []
        for s in SLOTS:
            q = S[s]
            zs = mk(q)
            o = np.argsort(-zs)
            c = np.cumsum(q["zdur"][o])
            zkeep = np.zeros(q["K"], bool)
            zkeep[o[c <= a.R1 * q["R"] * q["T"]]] = True
            inside = zkeep[q["z"]]
            idx = np.flatnonzero(inside)
            oo = idx[np.argsort(-q["p"][idx])]
            cc = np.cumsum(q["dur"][oo])
            keep = np.zeros(len(q["p"]), bool)
            keep[oo[cc <= q["R"] * q["T"]]] = True
            accs.append((q["dur"] * q["y"])[keep].sum() / q["dur"][keep].sum())
            aucs.append(auc(zs, (q["ztrue"] > q["R"]).astype(float), q["zdur"]))
        print(f"{name:<26}{np.mean(aucs):>9.3f}" +
              "".join(f"{x:>8.3f}" for x in accs) + f"{np.mean(accs):>9.3f}")
        return np.mean(aucs), np.mean(accs)

    # чистый блочный отбор — база
    accs = []
    for s in SLOTS:
        q = S[s]
        o = np.argsort(-q["p"]); c = np.cumsum(q["dur"][o])
        keep = np.zeros(len(q["p"]), bool); keep[o[c <= q["R"] * q["T"]]] = True
        accs.append((q["dur"] * q["y"])[keep].sum() / q["dur"][keep].sum())
    print(f"{'база (без зон)':<26}{'—':>9}" +
          "".join(f"{x:>8.3f}" for x in accs) + f"{np.mean(accs):>9.3f}\n")

    print("--- судья = смесь оракула с нашей оценкой")
    for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        run(f"beta={beta}", lambda q, b=beta: b * rank01(q["ztrue"]) +
            (1 - b) * rank01(q["zours"]))

    print("\n--- судья = оракул + шум (контроль)")
    for sig in (2.0, 1.0, 0.6, 0.4, 0.25, 0.15, 0.0):
        def mk(q, sg=sig):
            rng = np.random.default_rng(0)
            return rank01(q["ztrue"]) + rng.normal(0, sg, q["K"])
        run(f"шум sigma={sig}", mk)

    print("\n--- судья = наша оценка + новый источник качества q")
    print("    (моделируем: новый источник = оракул с AUC около 0,7-0,8)")
    for sig in (0.5, 0.35, 0.25):
        for wnew in (0.3, 0.5, 0.7):
            def mk(q, sg=sig, w=wnew):
                rng = np.random.default_rng(1)
                new = rank01(rank01(q["ztrue"]) + rng.normal(0, sg, q["K"]))
                return (1 - w) * rank01(q["zours"]) + w * new
            run(f"шум {sig} вес {wnew}", mk)


if __name__ == "__main__":
    main()
