#!/usr/bin/env python3
"""
Проверка: отличают ли теги звука решение монтажёра.

Пометка, которая не связана с решением, для модели вредна — она отвлекает
и создаёт ложную уверенность. Поэтому прежде чем показывать теги, меряем
по обучающим слотам: какова доля оставленного времени там, где тег сработал,
против общей доли. Порог тега задаём квантилем по стриму и перебираем его.
"""
import json
import numpy as np

DATASET = os.environ.get("SEM_DATASET", "work/dataset")
DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6"]


def main():
    for Q in (0.90, 0.95, 0.98):
        print(f"\n=== порог тега: верхние {(1-Q)*100:.0f} % окон стрима")
        print(f"{'тег':>16}{'минут':>9}{'доля эфира':>12}{'оставлено':>11}"
              f"{'база':>8}{'разница':>9}")
        print("-" * 65)
        acc = {}
        for s in SLOTS:
            B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
            cl = np.load(f"{DATA}/clap/clap_s{s}.npz", allow_pickle=True)
            C = cl["clap"].astype(np.float32)
            names = [str(x) for x in cl["names"]]
            hop = float(cl["clap_hop"])
            dur = np.array([b["dur"] for b in B])
            fr = np.array([b["used_frac"] for b in B])
            base = (dur * fr).sum() / dur.sum()
            for j, n in enumerate(names):
                thr = np.quantile(C[:, j], Q)
                hit = np.zeros(len(B), bool)
                for k, b in enumerate(B):
                    i0 = int(b["start"] / hop)
                    i1 = min(max(i0 + 1, int(np.ceil(b["end"] / hop))), len(C))
                    if i1 > i0 and C[i0:i1, j].max() >= thr:
                        hit[k] = True
                if hit.sum() == 0:
                    continue
                t = dur[hit].sum()
                kept = (dur * fr)[hit].sum() / t
                acc.setdefault(n, []).append((t, kept, base, t / dur.sum()))
        for n, v in sorted(acc.items(), key=lambda z: -abs(
                np.mean([x[1] for x in z[1]]) - np.mean([x[2] for x in z[1]]))):
            a = np.array(v)
            print(f"{n:>16}{a[:,0].sum()/60:>9.0f}{a[:,3].mean():>12.1%}"
                  f"{a[:,1].mean():>11.1%}{a[:,2].mean():>8.1%}"
                  f"{a[:,1].mean()-a[:,2].mean():>+9.1%}")


if __name__ == "__main__":
    main()
