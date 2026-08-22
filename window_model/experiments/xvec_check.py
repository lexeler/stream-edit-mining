#!/usr/bin/env python3
"""
Контроль: несут ли отпечатки голоса информацию о говорящем на уличном звуке?

Признак «изменчивость голоса» ухудшил модель. Прежде чем выбрасывать идею,
проверяем сам отпечаток. Если соседние по времени отпечатки похожи не сильнее
случайных пар — модель ловит шум, а не голос, и дальше идти незачем.

Заодно пробуем другую форму признака: расстояние до «хозяина эфира»
(усреднённый отпечаток по всей речи). Присутствие чужого голоса должно
давать всплеск, а это и есть «он с людьми».
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

    print(f"{'слот':>5}{'соседние':>10}{'через 5 мин':>13}{'случайные':>11}"
          f"{'разница':>10}")
    print("-" * 49)
    rng = np.random.default_rng(0)
    for s in sorted(by):
        z = np.load(f"{DATA}/xvec_s{s}.npz")
        xv = z["xv"].astype(np.float32)
        h = float(z["xv_hop"])
        # оставляем только окна, где есть речь
        rr = by[s]
        st = np.array([r["start"] for r in rr], np.float32)
        en = np.array([r["end"] for r in rr], np.float32)
        isw = np.array([r["is_word"] for r in rr], np.int8)
        t = np.arange(len(xv)) * h
        j = np.clip(np.searchsorted(st, t) - 1, 0, len(rr) - 1)
        sp = (isw[j] == 1) & (t < en[-1])
        V = xv[sp]
        if len(V) < 1000:
            continue
        adj = float((V[1:] * V[:-1]).sum(1).mean())
        k = int(300 / h)
        far = float((V[k:] * V[:-k]).sum(1).mean())
        i1 = rng.integers(0, len(V), 20000); i2 = rng.integers(0, len(V), 20000)
        rnd = float((V[i1] * V[i2]).sum(1).mean())
        print(f"{s:>5}{adj:>10.3f}{far:>13.3f}{rnd:>11.3f}{adj-rnd:>+10.3f}")

    print("\nЕсли 'соседние' заметно выше 'случайных' — отпечаток различает")
    print("голоса и участки. Если почти одинаково — это шум.\n")

    print("=== другая форма признака: расстояние до хозяина эфира ===")
    print(f"{'слот':>5}{'оставлено':>12}{'выброшено':>12}{'разница':>10}")
    print("-" * 39)
    for s in sorted(by):
        z = np.load(f"{DATA}/xvec_s{s}.npz")
        xv = z["xv"].astype(np.float32); h = float(z["xv_hop"])
        rr = by[s]
        st = np.array([r["start"] for r in rr], np.float32)
        isw = np.array([r["is_word"] for r in rr], np.int8)
        frac = np.array([r["used_frac"] for r in rr], np.float32)
        dur = np.array([r["dur"] for r in rr], np.float32)
        t = np.arange(len(xv)) * h
        j = np.clip(np.searchsorted(st, t) - 1, 0, len(rr) - 1)
        sp = isw[j] == 1
        host = xv[sp].mean(0); host /= np.linalg.norm(host) + 1e-6
        d = 1 - xv @ host
        # переносим на элементы
        e = np.clip((( (st + dur / 2) / h)).astype(int), 0, len(d) - 1)
        v = d[e]
        w = dur
        a = float((v * frac * w).sum() / (frac * w).sum())
        b = float((v * (1 - frac) * w).sum() / ((1 - frac) * w).sum())
        print(f"{s:>5}{a:>12.4f}{b:>12.4f}{a-b:>+10.4f}")


if __name__ == "__main__":
    main()
