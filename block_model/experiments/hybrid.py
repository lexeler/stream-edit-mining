#!/usr/bin/env python3
"""
Смесь двух источников: бустинг на форме речи и языковая модель на смысле.

Разбор контрастных границ показал, почему их стоит смешивать: внутри живой
сцены решение предсказывает ФОРМА речи (плотность знаков в секунду — 74,6 %
угадывания, какую из двух сторон выбросили), а тема не предсказывает ничего
(51,8 %, на трудном ядре — хуже случайного). Зато выбор самих сцен — это
смысл, и там языковая модель как раз в своей стихии.

Обе оценки переводим в ранги внутри стрима (шкалы у них несравнимые)
и смешиваем с весом.

  hybrid.py --slot 7 --tags v3s10 v2au
"""
import json, argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
W0 = 5.0     # шаг сетки, на которой считал бустинг


def ml_on_blocks(slot, blocks):
    d = np.load(f"{DATA}/pred_final_s{slot}.npy", allow_pickle=True).item()
    p = d["p"]
    t = (np.arange(len(p)) + 0.5) * W0
    out = np.zeros(len(blocks))
    for k, b in enumerate(blocks):
        m = (t >= b["start"]) & (t < b["end"])
        out[k] = p[m].mean() if m.any() else np.interp(
            (b["start"] + b["end"]) / 2, t, p)
    return out


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    blocks = S.load(a.slot)
    dur = np.array([b["dur"] for b in blocks])
    fr = np.array([b["used_frac"] for b in blocks])
    rate = (dur * fr).sum() / dur.sum()

    ml = rank01(ml_on_blocks(a.slot, blocks))
    print(f"слот {a.slot}: доля монтажёра {rate:.1%}, "
          f"случайная точность {rate:.3f}")
    m = S.metrics(blocks, ml >= np.quantile(ml, 1 - rate))
    print(f"\nтолько бустинг:        точность {m['точность']:.3f}")

    for tag in a.tags:
        d = json.load(open(f"{DATA}/ans/s{a.slot}_{tag}.json"))
        sc = {int(k): v for k, v in d["scores"].items()}
        llm = rank01(S.fill_pauses(blocks, sc))
        r = S.at_rate(a.slot, sc)
        print(f"\nтолько языковая [{tag}]: точность {r['точность']:.3f}")
        print(f"{'вес языковой':>14}{'точность':>10}{'полнота':>9}")
        for w in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
            mix = (1 - w) * ml + w * llm
            keep = np.zeros(len(mix), bool)
            o = np.argsort(-mix)
            c = np.cumsum(dur[o]) / dur.sum()
            keep[o[c <= rate]] = True
            mm = S.metrics(blocks, keep)
            print(f"{w:>14.2f}{mm['точность']:>10.3f}{mm['полнота']:>9.3f}")


if __name__ == "__main__":
    main()
