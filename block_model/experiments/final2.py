#!/usr/bin/env python3
"""
Итог: обучаемся на шести слотах, собираем выпуск для отложенного слота 2.

Конфигурация зафиксирована ДО того, как слот 2 вскрыт, и по нему ничего
не подбирается:
  блочная модель на репликах, 1159 признаков (форма речи, акустика, e5,
    CLIP, CLAP, описания кадров, типы от языковой модели, оценка прежней
    оконной модели), глубина 6, 800 итераций, 5 семян;
  смесь с прежней оконной моделью в отношении 0,7 к 0,3 — так вышло
    лучше всего на слоте 7;
  доля выпуска не подсказывается: берём полусумму средней предсказанной
    и «типичный выпуск канала / длина эфира», как в прежнем конвейере;
  сборка: Витерби со штрафом за переключение, резы притягиваются к тишине,
  выпуск начинается в первые полторы минуты.

  python3 final2.py --held 2
"""
import json, os, argparse, subprocess, sys
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier

RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
CUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SLOTS = ["1", "3", "4", "5", "6", "7"]
W0 = 5.0
MIX = 0.3          # вес блочной модели в смеси


def load(s):
    z = np.load(f"{RULES}/feat/s{s}.npz", allow_pickle=True)
    return z["X"], z["y"], z["dur"], z["start"]


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def prec_at(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool)
        k[o[c <= rate]] = True
        out.append(((dur * y)[k].sum() / dur[k].sum(),
                    (dur * y)[k].sum() / (dur * y).sum()))
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="2")
    a = ap.parse_args()
    tr = [s for s in SLOTS if s != a.held]

    Xtr = np.vstack([load(s)[0] for s in tr])
    ytr = np.concatenate([load(s)[1] for s in tr])
    wtr = np.concatenate([load(s)[2] for s in tr])
    X, y, dur, start = load(a.held)
    T = float(dur.sum())
    true_rate = float((dur * y).sum() / T)

    # классификация «войдёт / не войдёт» вместо регрессии на долю:
    # по шести обучающим слотам 0,621 против 0,615 у регрессии
    ps = []
    for sd in range(5):
        m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                               l2_leaf_reg=20, verbose=0, random_seed=sd,
                               thread_count=5, rsm=0.3, subsample=0.8,
                               bootstrap_type="Bernoulli")
        m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
        ps.append(m.predict_proba(X)[:, 1])
    pb = np.mean(ps, 0)

    # прежняя оконная модель на той же ленте блоков
    d0 = np.load(f"{CUT}/pred_final_s{a.held}.npy", allow_pickle=True).item()
    pw = d0["p"].astype(np.float64)
    tw = (np.arange(len(pw)) + 0.5) * W0
    mid = start + dur / 2
    pw_b = np.interp(mid, tw, pw)
    mix = (1 - MIX) * rank01(pw_b) + MIX * rank01(pb)

    # доля выпуска: модель решает сама, подсказки нет
    r_mean = float((np.interp(mid, tw, pw) * dur).sum() / T)
    typ = float(np.median([(load(s)[1] * load(s)[2]).sum() / 60 for s in tr]))
    r_dur = min(typ * 60 / T, 0.85)
    rate = (r_mean + r_dur) / 2

    print(f"=== слот {a.held}: эфир {T/60:.0f} мин, "
          f"монтажёр {(dur*y).sum()/60:.0f} мин ({true_rate:.1%})")
    print(f"типичный выпуск канала {typ:.0f} мин -> доля по хронометражу "
          f"{r_dur:.1%}; средняя предсказанная {r_mean:.1%}; берём {rate:.1%}")
    for nm, p in (("оконная", rank01(pw_b)), ("блочная", rank01(pb)),
                  ("смесь", mix)):
        a1, r1 = prec_at(y, dur, p, true_rate)
        print(f"  {nm:>8}: при длине монтажёра точность {a1:.3f} полнота {r1:.3f}")
    a2, r2 = prec_at(y, dur, mix, rate)
    print(f"  смесь при СВОЕЙ доле {rate:.1%} ({rate*T/60:.0f} мин): "
          f"точность {a2:.3f} полнота {r2:.3f}")
    print(f"  случайно: {true_rate:.3f}")

    np.save(f"{RULES}/final_s{a.held}.npy",
            {"mix": mix, "pb": pb, "pw": pw_b, "y": y, "dur": dur,
             "start": start, "rate": rate}, allow_pickle=True)
    print(f"\nсохранено: {RULES}/final_s{a.held}.npy")


if __name__ == "__main__":
    main()
