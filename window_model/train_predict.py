#!/usr/bin/env python3
"""
Итоговый конвейер: стрим -> монтажный лист.

Собрано из того, что проверено замерами:
  окна 5, 10, 20 и 30 с, предсказания усредняются (+0,018 к F1)
  доля выпуска = смесь средней предсказанной
    и «типичный выпуск канала / длина стрима»   (+0,016)
  сборка по правилам выпуска: начало в первые
    полторы минуты, резы в тишине, паузы внутри
    сцен не трогаем

  train_predict.py --slot 5            проверка на обучающем слоте
  train_predict.py --slot 2 --test     честный тест, вскрывать один раз

На выходе data/edl_sN.json — монтажный лист для рендера на Маке.
"""
import json, argparse, collections, os, sys
import numpy as np
from catboost import CatBoostRegressor
from windows import get_data, score, take_rate
from assemble import build, build_to_duration, evaluate

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
WINS = [5.0, 10.0, 20.0, 30.0]
SEEDS = 5


def to_grid(p, w, n, w0):
    t = (np.arange(len(p)) + 0.5) * w
    g = (np.arange(n) + 0.5) * w0
    return np.interp(g, t, p)


def predict_slot(slot, blocks, test, tr_slots=None):
    """предсказание по отложенному слоту, усреднение по окнам и семенам"""
    outs, D0 = [], None
    for w in WINS:
        Xtr, Dtr = get_data(w, blocks, "train")
        if test:
            Xte, Dte = get_data(w, blocks, "test")
            tr = sorted(Xtr); X, D = Xte[slot], Dte[slot]
        else:
            tr = [s for s in sorted(Xtr) if s != slot]
            X, D = Xtr[slot], Dtr[slot]
        A = np.vstack([Xtr[s] for s in tr])
        y = np.concatenate([Dtr[s][0] for s in tr])
        wt = np.concatenate([Dtr[s][1] for s in tr])
        ps = []
        for sd in range(SEEDS):
            m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                  l2_leaf_reg=20, verbose=0, random_seed=sd,
                                  thread_count=5, rsm=0.3, subsample=0.8,
                                  bootstrap_type="Bernoulli")
            m.fit(A, y, sample_weight=wt)
            ps.append(np.clip(m.predict(X), 0, 1))
        p = np.mean(ps, 0)
        if w == WINS[0]:
            D0 = D; outs.append(p); n0 = len(p)
        else:
            outs.append(to_grid(p, w, n0, WINS[0]))
        # типичный выпуск канала — только по обучающим слотам
        typ = float(np.median([(Dtr[s][0] * Dtr[s][1]).sum() / 60 for s in tr]))
    return np.mean(outs, 0), D0, typ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="5")
    ap.add_argument("--blocks", default="ac+pca+tag+vtag2+vid+vemb+xv+clap+llm+scene+nov2+surp+qn+icl")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    blocks = set(a.blocks.split("+"))

    p, D, typ = predict_slot(a.slot, blocks, a.test)
    Y, WT = D
    T = float(WT.sum())
    r_mean = float((p * WT).sum() / WT.sum())
    r_dur = min(typ * 60 / T, 0.85)
    rate = (r_mean + r_dur) / 2
    print(f"слот {a.slot}: эфир {T/60:.0f} мин, окон {len(p)}")
    print(f"типичный выпуск канала {typ:.0f} мин -> доля по хронометражу "
          f"{r_dur:.1%}; средняя предсказанная {r_mean:.1%}; берём {rate:.1%}")
    print(f"модель предлагает выпуск на {rate*T/60:.0f} мин, "
          f"у монтажёра {(Y*WT).sum()/60:.0f} мин ({(Y*WT).sum()/T:.1%})")
    np.save(f"{DATA}/pred_final_s{a.slot}.npy",
            {"p": p, "Y": Y, "WT": WT}, allow_pickle=True)
    o = np.argsort(-p); cc = np.cumsum(WT[o]) / T
    tr_min = float((Y * WT).sum())
    print(f"\n{'длина выпуска':>15}{'совпадение':>12}{'случайно':>10}")
    for r in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        kp = np.zeros(len(p), bool); kp[o[cc <= r]] = True
        ov = float((kp * Y * WT).sum()); pd = float((kp * WT).sum())
        print(f"{pd/60:>12.0f} мин{ov/pd:>12.3f}{tr_min/T:>10.3f}")
    s0 = score(take_rate(p, WT, rate), Y, WT)
    print(f"по окнам:  точн {s0[0]:.3f}  полн {s0[1]:.3f}  F1 {s0[2]:.3f}  "
          f"IoU {s0[3]:.3f}")

    rows = [json.loads(l) for l in
            open(f"{DATA}/{'test' if a.test else 'train'}.jsonl")]
    if a.test:
        ans = [json.loads(l) for l in open(f"{DATA}/test_answers.jsonl")]
        for r, q in zip(rows, ans):
            r["used_frac"] = q["used_frac"]
    rows = [r for r in rows if r["slot"] == a.slot]

    # огибающая громкости, чтобы резы попадали в тишину: монтажёр почти
    # всегда режет в паузе (склеек «речь|речь» всего 1,5 %)
    rms = None
    fl = os.path.join(os.environ.get("SEM_WORK", "work/window"), f"aud/s{a.slot}.flac")
    if os.path.exists(fl):
        import subprocess
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", fl,
                              "-f", "s16le", "-ac", "1", "-ar", "8000", "-"],
                             capture_output=True).stdout
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        W = 80                                  # окно 10 мс при 8 кГц
        n = len(x) // W
        rms = np.sqrt((x[:n * W].reshape(n, W) ** 2).mean(1))
        print(f"огибающая громкости: {n} точек, шаг 10 мс")

    print(f"\n{'штраф':>7}{'пауза':>7}{'кусков':>8}{'минут':>8}{'медиана с':>11}"
          f"{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}")
    print("-" * 73)
    best = None
    for lam in (0.0, 0.05, 0.1, 0.2):
        for pd in (8.0, 3.0, 2.0, 1.5):
            segs, _ = build_to_duration(rows, p, WINS[0], rate * T, lam,
                                        rms=rms, pause_drop=pd)
            m = evaluate(segs, rows)
            ln = [b - x for x, b in segs] or [0]
            print(f"{lam:>7.2f}{pd:>7.1f}{m['кусков']:>8}{m['минут']:>8.1f}"
                  f"{np.median(ln):>11.1f}{m['точность']:>8.3f}{m['полнота']:>8.3f}"
                  f"{m['F1']:>8.3f}{m['IoU']:>8.3f}")
            if abs(lam - a.lam) < 1e-9 and abs(pd - a.pause) < 1e-9:
                best = (segs, m)
    segs, m = best
    out = f"{DATA}/edl_s{a.slot}.json"
    json.dump({"slot": a.slot, "rate": rate, "segments": segs, "metrics": m},
              open(out, "w"), ensure_ascii=False)
    nseg = sum(1 for i in range(1, len(rows))
               if (rows[i]["used_frac"] > .5) != (rows[i - 1]["used_frac"] > .5)) // 2
    print(f"\nмонтажный лист сохранён: {out}")
    print(f"у монтажёра кусков {nseg}, у модели {m['кусков']}")


if __name__ == "__main__":
    main()
