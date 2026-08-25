#!/usr/bin/env python3
"""
Итоговая модель третьего захода: блочная + движение в кадре.

Отличия от final2.py, каждое обосновано кросс-проверкой по шести слотам:
  * без смеси с оконной моделью — чистая блочная лучше (0,623 против 0,620);
  * добавлены признаки движения в кадре (+0,005) — они были в оконной
    модели, но в блочную не попадали;
  * доля выпуска считается как раньше: полусумма средней предсказанной
    и «типичный выпуск канала / длина эфира». Подсказки хронометража нет.

  python3 train_predict.py --held 2 [--motion2]
"""
import os, argparse
import numpy as np
from catboost import CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def load(s, which):
    f = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
    X = f["X"]
    for sub in which:
        p = f"{RULES}/{sub}/s{s}.npy"
        if os.path.exists(p):
            X = np.hstack([X, np.load(p, allow_pickle=True).item()["X"]])
    return X.astype(np.float32), f["y"], f["dur"], f["start"]


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append(((dur * y)[k].sum() / dur[k].sum(),
                    (dur * y)[k].sum() / (dur * y).sum()))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="2")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--motion2", action="store_true",
                    help="взять подробное движение вместо простого")
    ap.add_argument("--clean", action="store_true",
                    help="убрать из обучения блоки, задетые резом частично "
                         "(+0,004 в одиночку, но с движением не складывается)")
    ap.add_argument("--wb", type=float, default=0.7,
                    help="вес блочной модели в смеси с оконной. Прежний "
                         "конвейер держал 0,3 — это оказалось наоборот: "
                         "на шести слотах 0,5–0,85 дают 0,626, а 0,3 — 0,620")
    a = ap.parse_args()
    which = ["motion2f"] if a.motion2 else ["motion"]
    tr = [s for s in SLOTS if s != a.held]

    # Чистка обучающей выборки от блоков, задетых резом частично, сама по себе
    # даёт +0,004, но С ДВИЖЕНИЕМ НЕ СКЛАДЫВАЕТСЯ: вместе +0,003 против +0,005
    # у одного движения. Поэтому по умолчанию выключена.
    Xs, ys, ws = [], [], []
    for s in tr:
        Xi, yi, wi, _ = load(s, which)
        keep = (~((yi > 0.2) & (yi < 0.8)) if a.clean
                else np.ones(len(yi), bool))
        Xs.append(Xi[keep]); ys.append((yi[keep] > 0.5).astype(int))
        ws.append(wi[keep])
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    wtr = np.concatenate(ws)
    X, y, dur, start = load(a.held, which)
    T = float(dur.sum())
    true_rate = float((dur * y).sum() / T)

    ps = []
    for sd in range(a.seeds):
        m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                               l2_leaf_reg=20, verbose=0, random_seed=sd,
                               thread_count=5, rsm=0.3, subsample=0.8,
                               bootstrap_type="Bernoulli")
        m.fit(Xtr, ytr, sample_weight=wtr)
        ps.append(m.predict_proba(X)[:, 1])
    pb = rank01(np.mean(ps, 0))

    # оконная модель прежнего конвейера, снесённая на ленту блоков
    d0 = np.load(os.path.join(os.environ.get("SEM_WORK", "work/window"), f"data/pred_final_s{a.held}.npy"),
                 allow_pickle=True).item()
    pw = d0["p"].astype(np.float64)
    tw = (np.arange(len(pw)) + 0.5) * 5.0
    mid = start + dur / 2
    pw_b = rank01(np.interp(mid, tw, pw))
    mix = a.wb * pb + (1 - a.wb) * pw_b

    r_mean = float((np.mean(ps, 0) * dur).sum() / T)
    typ = float(np.median([(load(s, which)[1] * load(s, which)[2]).sum() / 60
                           for s in tr]))
    rate = (r_mean + min(typ * 60 / T, 0.85)) / 2

    print(f"=== слот {a.held}: эфир {T/60:.0f} мин, монтажёр "
          f"{(dur*y).sum()/60:.0f} мин ({true_rate:.1%}), признаков {X.shape[1]}")
    for nm, p in (("оконная", pw_b), ("блочная", pb),
                  (f"смесь {a.wb:.1f}/{1-a.wb:.1f}", mix)):
        acc, rec = prec(y, dur, p, true_rate)
        print(f"  {nm:>14}: при длине монтажёра точность {acc:.3f} "
              f"полнота {rec:.3f}")
    acc2, rec2 = prec(y, dur, mix, rate)
    print(f"  смесь при своей доле {rate:.1%} ({rate*T/60:.0f} мин): "
          f"точность {acc2:.3f} полнота {rec2:.3f}")
    print(f"  случайно: {true_rate:.3f}")

    np.save(f"{RULES}/final3_s{a.held}.npy",
            {"pb": mix, "mix": mix, "pw": pw_b, "y": y, "dur": dur,
             "start": start, "rate": rate}, allow_pickle=True)
    print(f"сохранено: {RULES}/final3_s{a.held}.npy")


if __name__ == "__main__":
    main()
