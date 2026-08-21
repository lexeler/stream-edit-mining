#!/usr/bin/env python3
"""
Микро-правило: что монтажёр выбрасывает ВНУТРИ оставленных сцен.

Макро-модель решает, какие области эфира выживают. Но внутри них монтажёр
делает ещё около 280 мелких резов на стрим. Если их не повторить, нарезка
будет выглядеть блоками, а не его работой.

Берём выброшенные куски короче 5 с, у которых с обеих сторон оставленное,
и смотрим: из чего они состоят и как вероятность выброса зависит
от длительности паузы.
"""
import os
import json, collections
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)

    # из чего состоят короткие выбросы внутри оставленного
    tot_sil = tot_wrd = 0.0
    n_runs = 0
    lens = []
    for s in sorted(by):
        rr = by[s]
        keep = np.array([r["used_frac"] for r in rr]) > 0.5
        dur = np.array([r["dur"] for r in rr], np.float32)
        isw = np.array([r["is_word"] for r in rr], np.int8)
        k = 0
        while k < len(keep):
            if keep[k]:
                k += 1; continue
            j = k
            while j < len(keep) and not keep[j]:
                j += 1
            L = dur[k:j].sum()
            if L <= 5.0 and k > 0 and j < len(keep):
                n_runs += 1; lens.append(L)
                tot_sil += dur[k:j][isw[k:j] == 0].sum()
                tot_wrd += dur[k:j][isw[k:j] == 1].sum()
            k = j
    print(f"коротких выбросов внутри оставленного: {n_runs}, "
          f"суммарно {(tot_sil+tot_wrd)/60:.1f} мин")
    print(f"  из них тишина {tot_sil/(tot_sil+tot_wrd)*100:.1f} %, "
          f"речь {tot_wrd/(tot_sil+tot_wrd)*100:.1f} %")
    print(f"  медиана длины {np.median(lens):.2f} с")

    # вероятность выброса паузы в зависимости от её длины,
    # только там, где соседи оставлены
    print("\n=== пауза внутри оставленной сцены: выбросят или нет ===")
    print(f"{'длина паузы':>14}{'штук':>8}{'выброшено':>12}")
    print("-" * 34)
    buck = [(0, .3), (.3, .5), (.5, .8), (.8, 1.2), (1.2, 2), (2, 3), (3, 5),
            (5, 10), (10, 1e9)]
    cnt = collections.defaultdict(lambda: [0, 0])
    for s in sorted(by):
        rr = by[s]
        keep = np.array([r["used_frac"] for r in rr]) > 0.5
        for i in range(1, len(rr) - 1):
            if rr[i]["is_word"]:
                continue
            # ищем ближайших соседей-слов слева и справа
            a = i - 1
            while a > 0 and not rr[a]["is_word"]:
                a -= 1
            b = i + 1
            while b < len(rr) - 1 and not rr[b]["is_word"]:
                b += 1
            if not (keep[a] and keep[b]):
                continue
            d = rr[i]["dur"]
            for lo, hi in buck:
                if lo <= d < hi:
                    cnt[(lo, hi)][0] += 1
                    cnt[(lo, hi)][1] += int(not keep[i])
                    break
    for lo, hi in buck:
        n, k = cnt[(lo, hi)]
        if n:
            name = f"{lo:g}-{hi:g} с" if hi < 1e8 else f">{lo:g} с"
            print(f"{name:>14}{n:>8}{k/n*100:>11.1f}%")

    print("\nЧитать так: с какой длины пауза почти наверняка вырезается —")
    print("это и есть порог микро-правила для сборки.")


if __name__ == "__main__":
    main()
