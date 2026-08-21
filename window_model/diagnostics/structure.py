#!/usr/bin/env python3
"""
Из чего состоит работа монтажёра: микро-резка пауз или макро-выброс кусков?

Разбираем ленту на непрерывные участки "оставлено" и "выброшено" и смотрим,
сколько времени уходит на короткие технические выбросы, а сколько на длинные
содержательные. От этого зависит, что вообще имеет смысл предсказывать.

Только обучающие слоты. Слот 2 не трогаем.
"""
import os
import json, collections
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
BUCK = [(0, 0.5), (0.5, 2), (2, 5), (5, 15), (15, 60), (60, 300), (300, 1e9)]


def runs(frac, dur):
    """непрерывные участки одного решения -> список (оставлено?, секунды)"""
    keep = frac > 0.5
    out, cur, acc = [], keep[0], 0.0
    for k in range(len(keep)):
        if keep[k] != cur:
            out.append((cur, acc)); cur, acc = keep[k], 0.0
        acc += dur[k]
    out.append((cur, acc))
    return out


def table(rs, title):
    print(f"\n  {title}")
    print(f"    {'длина':>12}{'штук':>8}{'всего с':>10}{'доля времени':>14}")
    tot = sum(d for d in rs)
    for lo, hi in BUCK:
        sel = [d for d in rs if lo <= d < hi]
        if not sel:
            continue
        name = f"{lo:g}-{hi:g}с" if hi < 1e8 else f">{lo:g}с"
        print(f"    {name:>12}{len(sel):>8}{sum(sel):>10.0f}"
              f"{sum(sel)/tot*100:>13.1f}%")
    print(f"    {'ИТОГО':>12}{len(rs):>8}{tot:>10.0f}")
    print(f"    медиана {np.median(rs):.2f} с, среднее {np.mean(rs):.2f} с")


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)

    all_keep, all_drop = [], []
    print("=== баланс по слотам ===")
    print(f"{'слот':>5}{'эфир мин':>10}{'выпуск мин':>12}{'доля':>8}"
          f"{'речь %':>9}{'речь в выпуске %':>18}")
    print("-" * 62)
    for s in sorted(by):
        rr = by[s]
        dur = np.array([r["dur"] for r in rr], np.float32)
        frac = np.array([r["used_frac"] for r in rr], np.float32)
        isw = np.array([r["is_word"] for r in rr], np.int8)
        T = dur.sum(); K = (dur * frac).sum()
        sp = dur[isw == 1].sum()
        kept_speech = (dur * frac)[isw == 1].sum()
        print(f"{s:>5}{T/60:>10.1f}{K/60:>12.1f}{K/T*100:>7.1f}%"
              f"{sp/T*100:>8.1f}%{kept_speech/K*100:>17.1f}%")
        rs = runs(frac, dur)
        all_keep += [d for k, d in rs if k]
        all_drop += [d for k, d in rs if not k]

    print("\n=== структура решений (все обучающие слоты вместе) ===")
    table(all_keep, "УЧАСТКИ ОСТАВЛЕНО")
    table(all_drop, "УЧАСТКИ ВЫБРОШЕНО")

    print("\nВывод читать так: если большая часть выброшенного времени лежит")
    print("в длинных кусках, то главное — макро-решения, а не резка пауз.")


if __name__ == "__main__":
    main()
