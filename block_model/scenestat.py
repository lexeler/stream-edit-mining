#!/usr/bin/env python3
"""
Годятся ли сцены как единица решения.

Сцена — кусок эфира между длинными паузами. Если монтажёр решает сценами
целиком, можно давать модели список сцен и просить выбрать лучшие: это
даёт ей взгляд на весь стрим сразу и позволяет сравнивать. Если же он
режет внутри сцены, потолок такого отбора низкий и надо решать по блокам.

Меряем при разных порогах паузы:
  сколько сцен, какой длины;
  однородность: доля сцен, где решение почти единообразно (>80 % в одну сторону);
  ПОТОЛОК отбора целыми сценами — точность, если брать сцены с долей > 0,5.
"""
import json, glob, os
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")


def scenes(blocks, gap):
    sc, cur = [], []
    for b in blocks:
        if b["kind"] == "pause" and b["dur"] >= gap and cur:
            sc.append(cur); cur = []
        cur.append(b)
    if cur:
        sc.append(cur)
    return sc


def main():
    slots = ["1", "3", "4", "5", "6"]
    print(f"{'пауза':>6}{'сцен':>7}{'медиана с':>11}{'однородных':>12}"
          f"{'потолок точн':>14}{'потолок полн':>14}{'доля':>7}")
    print("-" * 72)
    for gap in (6.0, 10.0, 15.0, 25.0, 40.0):
        rows = []
        for s in slots:
            B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
            sc = scenes(B, gap)
            durs, fracs = [], []
            for g in sc:
                d = sum(b["dur"] for b in g)
                f = sum(b["dur"] * b["used_frac"] for b in g) / d if d else 0
                durs.append(d); fracs.append(f)
            durs = np.array(durs); fracs = np.array(fracs)
            homo = ((fracs > 0.8) | (fracs < 0.2)).mean()
            keep = fracs > 0.5
            ov = (durs * fracs)[keep].sum()
            pred = durs[keep].sum()
            true = (durs * fracs).sum()
            rows.append((len(sc), np.median(durs), homo, ov / pred if pred else 0,
                         ov / true, pred / durs.sum()))
        a = np.array(rows)
        print(f"{gap:>6.0f}{a[:,0].mean():>7.0f}{a[:,1].mean():>11.0f}"
              f"{a[:,2].mean():>11.0%}{a[:,3].mean():>14.3f}"
              f"{a[:,4].mean():>14.3f}{a[:,5].mean():>7.1%}")

    print("\nДля сравнения: потолок поблочной разбивки — точность 0,87.")
    print("\nРаспределение сцен по однородности (пауза 15 с, все слоты):")
    allf, alld = [], []
    for s in slots:
        B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
        for g in scenes(B, 15.0):
            d = sum(b["dur"] for b in g)
            if d < 5:
                continue
            f = sum(b["dur"] * b["used_frac"] for b in g) / d
            allf.append(f); alld.append(d)
    allf = np.array(allf); alld = np.array(alld)
    for lo, hi, name in ((0, 0.1, "почти всё вырезано"), (0.1, 0.35, "вырезано больше"),
                         (0.35, 0.65, "пополам"), (0.65, 0.9, "оставлено больше"),
                         (0.9, 1.01, "почти всё оставлено")):
        m = (allf >= lo) & (allf < hi)
        print(f"  {name:>22}: сцен {m.sum():>4} ({m.mean():>5.1%}), "
              f"времени {alld[m].sum()/60:>6.0f} мин ({alld[m].sum()/alld.sum():>5.1%})")


if __name__ == "__main__":
    main()
