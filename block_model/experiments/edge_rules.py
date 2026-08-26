#!/usr/bin/env python3
"""
Правила краёв, найденные разбором пяти выпусков, — и проверка их прироста.

Из разбора концовок:
  * выпуск заканчивается НА ПОСЛЕДНЕЙ реплике эфира — 5 эфиров из 5,
    после неё остаётся только тишина 3–17 секунд;
  * оф («спасибо всем, я офаю, завтра стрим, всем пока») — монолог 17–30 с
    в последних 0,8 % эфира, взят на 0,93–1,00;
  * пауза между двумя выброшенными кусками в хвосте не выживает никогда
    (0,0 % из 39 минут).

Из разбора начал:
  * первые 30 секунд эфира выживают на 1,2 %, первая минута — на 2,9 %;
  * первый кусок выпуска начинается в окне 0:26–1:23.

Здесь эти правила накладываются на оценку модели и меряется, что меняется.

  edge_rules.py                — проверка по шести обучающим слотам
"""
import os
import json, argparse
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SLOTS = ["1", "3", "4", "5", "6", "7"]


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def apply_rules(p, dur, start, kind, text, head=True, tail=True):
    """возвращает изменённую оценку"""
    p = p.copy()
    T = float(start[-1] + dur[-1])
    if head:
        # первые полминуты эфира почти не выживают: 1,2 %
        p[start + dur <= 30] = 0.0
        # до 0:26 не выживает вообще ничего ни на одном эфире
        p[start + dur <= 26] = -0.1
    if tail:
        # финал выпуска — последняя речь эфира. Находим последнюю реплику
        # и поднимаем её и весь оф перед ней на самый верх
        sp = np.flatnonzero(kind == 1)
        if len(sp):
            last = sp[-1]
            # оф: монолог в последних 0,8 % эфира
            of_from = T * 0.992
            m = (start >= of_from) & (kind == 1)
            if m.any():
                p[m] = 1.0
            p[last] = 1.0
    return p


def prec(y, dur, p, rate, seeds=7):
    rng = np.random.default_rng(0); out = []
    for _ in range(seeds):
        o = np.argsort(-(p + rng.random(len(p)) * 1e-9))
        c = np.cumsum(dur[o]) / dur.sum()
        k = np.zeros(len(p), bool); k[o[c <= rate]] = True
        out.append((dur * y)[k].sum() / dur[k].sum())
    return float(np.mean(out))


def main():
    print(f"{'слот':>5}{'как есть':>10}{'+ начало':>10}{'+ конец':>10}{'+ оба':>10}")
    agg = []
    for s in SLOTS:
        B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
        d = np.load(f"{DATA}/final_s{s}.npy", allow_pickle=True).item()
        p, y, dur, start = rank01(d["mix"]), d["y"], d["dur"], d["start"]
        kind = np.array([1 if b["kind"] == "speech" else 0 for b in B])
        text = [b["text"] for b in B]
        rate = float((dur * y).sum() / dur.sum())
        base = prec(y, dur, p, rate)
        h = prec(y, dur, apply_rules(p, dur, start, kind, text, True, False), rate)
        t = prec(y, dur, apply_rules(p, dur, start, kind, text, False, True), rate)
        b = prec(y, dur, apply_rules(p, dur, start, kind, text, True, True), rate)
        agg.append((base, h, t, b))
        print(f"{s:>5}{base:>10.3f}{h:>10.3f}{t:>10.3f}{b:>10.3f}")
    a = np.array(agg)
    print(f"{'средн':>5}{a[:,0].mean():>10.3f}{a[:,1].mean():>10.3f}"
          f"{a[:,2].mean():>10.3f}{a[:,3].mean():>10.3f}")


if __name__ == "__main__":
    main()
