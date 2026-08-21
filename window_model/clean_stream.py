#!/usr/bin/env python3
"""
Готовая чистка стрима: убрать мусор, отдать остальное.

Собрано из проверенного:
  классификация с уравненными классами (лучше переворота регрессии
    на 0,010–0,014 при выбросе 10–30 %);
  два захода — выбросили уверенное, переучились на остатке, выбросили ещё
    (ещё +0,012);
  усреднение по окнам 5, 10, 20, 30 с.

На выходе — монтажный лист того, что ОСТАЁТСЯ, и честные цифры:
сколько убрано, какая доля убранного действительно лишняя, сколько
хорошего материала потеряно.

  clean_stream.py --slot 5              проверка на обучающем слоте
  clean_stream.py --slot 2 --test       честный тест
"""
import json, argparse, os, sys
import numpy as np
from catboost import CatBoostClassifier
from windows import get_data

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
BLOCKS = "ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn+icl"
WINS = [5.0, 10.0, 20.0, 30.0]
SEEDS = 3


def to_grid(p, w, n, w0):
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n) + 0.5) * w0
    return np.interp(g, t, p)


def fit_pred(A, y, w, Xte, seeds=SEEDS):
    ps = []
    for sd in range(seeds):
        m = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                               l2_leaf_reg=20, verbose=0, random_seed=sd,
                               thread_count=5, rsm=0.3, subsample=0.8,
                               bootstrap_type="Bernoulli",
                               auto_class_weights="Balanced")
        m.fit(A, (y > 0.5).astype(int), sample_weight=w)
        ps.append(m.predict_proba(Xte)[:, 1])
    return np.mean(ps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--frac", type=float, default=0.20,
                    help="какую долю эфира убрать")
    ap.add_argument("--two", action="store_true",
                    help="два захода: на честном тесте оказалось ХУЖЕ")
    a = ap.parse_args()
    blocks = set(BLOCKS.split("+"))

    # первый заход: усреднение по четырём окнам
    P, D0, Xb = [], None, {}
    for w in WINS:
        Xtr, Dtr = get_data(w, blocks, "train")
        if a.test:
            Xte, Dte = get_data(w, blocks, "test")
            tr = sorted(Xtr); X, D = Xte[a.slot], Dte[a.slot]
        else:
            tr = [s for s in sorted(Xtr) if s != a.slot]
            X, D = Xtr[a.slot], Dtr[a.slot]
        p = fit_pred(np.vstack([Xtr[s] for s in tr]),
                     np.concatenate([1 - Dtr[s][0] for s in tr]),
                     np.concatenate([Dtr[s][1] for s in tr]), X)
        if w == WINS[0]:
            D0 = D; n0 = len(p); P.append(p)
            Xb = {"X": X, "tr": tr, "Xtr": Xtr, "Dtr": Dtr}
        else:
            P.append(to_grid(p, w, n0, WINS[0]))
    p1 = np.mean(P, 0)
    Y, WT = D0
    T = float(WT.sum())

    def cut(order, frac, base=None):
        kill = np.zeros(len(Y), bool) if base is None else base.copy()
        c = np.cumsum(WT[order]) / T
        kill[order[c <= frac]] = True
        return kill

    o1 = np.argsort(-p1)
    kill1 = cut(o1, a.frac)

    kill = kill1
    # второй заход отключён по умолчанию: на перекрёстной проверке он давал
    # +0,012, а на честном тесте провалился (0,781 против 0,817 при 25 %)
    if a.two:
        Xtr, Dtr, tr = Xb["Xtr"], Xb["Dtr"], Xb["tr"]
        A, yy, ww = [], [], []
        for s in tr:
            ps = fit_pred(np.vstack([Xtr[t] for t in tr if t != s]),
                          np.concatenate([1 - Dtr[t][0] for t in tr if t != s]),
                          np.concatenate([Dtr[t][1] for t in tr if t != s]),
                          Xtr[s])
            oo = np.argsort(-ps)
            cc = np.cumsum(Dtr[s][1][oo]) / Dtr[s][1].sum()
            k1 = np.zeros(len(ps), bool); k1[oo[cc <= a.frac]] = True
            m = ~k1
            A.append(Xtr[s][m]); yy.append(1 - Dtr[s][0][m])
            ww.append(Dtr[s][1][m])
        rest = ~kill1
        p2 = fit_pred(np.vstack(A), np.concatenate(yy), np.concatenate(ww),
                      Xb["X"][rest])
        idx = np.flatnonzero(rest)[np.argsort(-p2)]
        c2 = np.cumsum(WT[idx]) / T
        kill = kill1.copy(); kill[idx[c2 <= 0.10]] = True

    base = float(((1 - Y) * WT).sum() / T)
    killed = float((kill * WT).sum())
    right = float((kill * (1 - Y) * WT).sum())
    lost = float((kill * Y * WT).sum())
    print(f"слот {a.slot}: эфир {T/60:.0f} мин, монтажёр оставил "
          f"{(Y*WT).sum()/60:.0f} мин ({1-base:.1%})")
    print(f"\nубрано {killed/60:.0f} мин ({killed/T:.0%} эфира)")
    print(f"  из них действительно лишнее: {right/killed:.3f}  "
          f"(случайный выброс дал бы {base:.3f})")
    print(f"  потеряно хорошего материала: {lost/60:.1f} мин "
          f"({lost/(Y*WT).sum():.1%} от того, что монтажёр оставил)")
    print(f"  монтажёру остаётся: {(T-killed)/60:.0f} мин вместо {T/60:.0f}")

    print(f"\n{'убрать':>9}{'верно':>9}{'случайно':>10}{'потеряно, мин':>15}"
          f"{'остаток, мин':>14}")
    print("-" * 58)
    for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        k = cut(o1, f)
        kd = float((k * WT).sum()); r = float((k * (1 - Y) * WT).sum())
        ls = float((k * Y * WT).sum())
        print(f"{f:>8.0%}{r/kd:>9.3f}{base:>10.3f}{ls/60:>15.1f}"
              f"{(T-kd)/60:>14.0f}")

    # монтажный лист того, что остаётся
    rows = [json.loads(l) for l in
            open(f"{DATA}/{'test' if a.test else 'train'}.jsonl")]
    rows = [r for r in rows if r["slot"] == a.slot]
    st = np.array([r["start"] for r in rows], np.float32)
    dur = np.array([r["dur"] for r in rows], np.float32)
    wid = np.clip(((st + dur / 2) / WINS[0]).astype(int), 0, len(kill) - 1)
    keep_el = ~kill[wid]
    segs = []
    for i in range(len(rows)):
        if not keep_el[i]:
            continue
        aa, bb = float(st[i]), float(st[i] + dur[i])
        if segs and aa - segs[-1][1] < 1e-6:
            segs[-1][1] = bb
        else:
            segs.append([aa, bb])
    m = []
    for aa, bb in segs:
        if m and aa - m[-1][1] <= 0.5:
            m[-1][1] = bb
        else:
            m.append([aa, bb])
    m = [s for s in m if s[1] - s[0] >= 1.5]
    out = f"{DATA}/clean_s{a.slot}.json"
    json.dump({"slot": a.slot, "segments": [[round(x, 3), round(y, 3)]
                                            for x, y in m],
               "минут": round(sum(y - x for x, y in m) / 60, 1)},
              open(out, "w"), ensure_ascii=False)
    print(f"\nлист очищенного стрима: {out} — {len(m)} кусков, "
          f"{sum(y-x for x,y in m)/60:.0f} мин")


if __name__ == "__main__":
    main()
