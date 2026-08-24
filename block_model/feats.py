#!/usr/bin/env python3
"""
Признаки на блок-реплику для обучаемой модели.

Что кладём и почему:
  форма речи — на парном контроле границ плотность знаков в секунду даёт
    74,6 % угадывания, какую из двух сторон монтажёр выбросил, длина блока
    70 %, а словарь тем — 51,8 %, то есть ничего. Значит форма важнее темы;
  окрестность — решение зависит не от реплики, а от зоны: длина цепочки
    подготовки, есть ли результат рядом, насколько плотно идёт речь вокруг;
  типы от языковой модели — то, чего нет ни в звуке, ни в форме: едут,
    ждут, обслуживают чат, играют. Веса типов считает модель по данным;
  новизна — было ли такое в этом же стриме раньше (повтор монтажёр режет).

  feats.py --slot 7 --marks m1
Кладёт data/feat/s<N>.npz
"""
import json, os, re, argparse, math
from collections import Counter
import numpy as np

DATASET = os.environ.get("SEM_DATASET", "work/dataset")
DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SRC = {
    "1": "slot1_2026-07-30_futbol-mozgi.labeled.json",
    "2": "slot2_2026-07-31_gulyanka.json",
    "3": "slot3_2026-08-03_tarzanka-zabroshka.labeled.json",
    "4": "slot4_2026-08-04_razvlekalovka.labeled.json",
    "5": "slot5_2026-06-12_matiz.labeled.json",
    "6": "slot6_2026-06-30_2mp4.labeled.json",
    "7": "slot7_2026-06-30_varenikovskiy.labeled.json",
}
TYPES = list("AGRPDWTSFX")
FLAGS = list("!rn")
CLAP_SHOW = ["смех", "машина", "телефон", "музыка", "еда", "тишина", "улица",
             "ветер", "вода", "толпа", "крик", "разговор", "один", "шаги"]


def roll(vals, times, dur, sec):
    """среднее значение признака по окну ±sec секунд (по времени, не по блокам)"""
    n = len(vals)
    out = np.zeros(n)
    c = np.concatenate([[0.0], np.cumsum(vals * dur)])
    cd = np.concatenate([[0.0], np.cumsum(dur)])
    for i in range(n):
        lo = np.searchsorted(times, times[i] - sec)
        hi = np.searchsorted(times, times[i] + sec)
        d = cd[hi] - cd[lo]
        out[i] = (c[hi] - c[lo]) / d if d > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--marks", default="m1")
    a = ap.parse_args()
    s = a.slot
    B = json.load(open(f"{DATA}/s{s}.blocks.json"))["blocks"]
    items = json.load(open(f"{DATASET}/{SRC[s]}"))["items"]

    n = len(B)
    dur = np.array([b["dur"] for b in B])
    start = np.array([b["start"] for b in B])
    mid = start + dur / 2
    T = float(start[-1] + dur[-1])
    is_sp = np.array([b["kind"] == "speech" for b in B], float)
    text = [b["text"] for b in B]
    nchar = np.array([len(t) for t in text], float)
    nword = np.array([len(t.split()) for t in text], float)

    # ---- форма речи
    dens_char = nchar / np.maximum(dur, 0.3)          # знаков в секунду
    dens_word = nword / np.maximum(dur, 0.3)
    wlen = nchar / np.maximum(nword, 1)
    # пауза перед блоком: длительность предыдущего блока-тишины
    pause_before = np.zeros(n)
    for i in range(1, n):
        pause_before[i] = dur[i - 1] if B[i - 1]["kind"] == "pause" else 0.0
    pause_after = np.zeros(n)
    for i in range(n - 1):
        pause_after[i] = dur[i + 1] if B[i + 1]["kind"] == "pause" else 0.0

    # ---- акустика из ленты элементов
    st = np.array([r["start"] for r in items]); en = np.array([r["end"] for r in items])
    vol = np.array([r["f"]["vol"] for r in items])
    isw = np.array([r["f"]["is_word"] for r in items])
    bvol = np.full(n, np.nan)
    for k in range(n):
        m = (st < B[k]["end"]) & (en > B[k]["start"]) & (isw == 1)
        if m.any():
            bvol[k] = vol[m].mean()
    med = np.nanmedian(bvol)
    bvol = np.where(np.isnan(bvol), med, bvol)
    vol_rel = (bvol - med) / (np.nanstd(bvol) + 1e-6)

    # ---- теги CLAP числом
    cl = np.load(f"{DATA}/clap/clap_s{s}.npz", allow_pickle=True)
    C = cl["clap"].astype(np.float32)
    names = [str(x) for x in cl["names"]]
    hop = float(cl["clap_hop"])
    clap = np.zeros((n, len(CLAP_SHOW)))
    for j, nm in enumerate(CLAP_SHOW):
        if nm not in names:
            continue
        col = names.index(nm)
        q = np.quantile(C[:, col], 0.5)
        sd = C[:, col].std() + 1e-6
        for k in range(n):
            i0 = int(B[k]["start"] / hop)
            i1 = min(max(i0 + 1, int(math.ceil(B[k]["end"] / hop))), len(C))
            if i1 > i0:
                clap[k, j] = (C[i0:i1, col].mean() - q) / sd

    # ---- типы от языковой модели
    mk_path = f"{DATA}/marks/s{s}_{a.marks}.json"
    TY = np.zeros((n, len(TYPES)))
    FL = np.zeros((n, len(FLAGS)))
    have_marks = os.path.exists(mk_path)
    marks = {}
    if have_marks:
        marks = {int(k): v for k, v in json.load(open(mk_path))["marks"].items()}
        # тишине наследуем тип соседа слева
        last = None
        for i in range(n):
            v = marks.get(i)
            if v is None and last is not None and B[i]["kind"] == "pause":
                v = last
            if v:
                last = v
                TY[i, TYPES.index(v[0])] = 1
                for f in v[1:]:
                    if f in FLAGS:
                        FL[i, FLAGS.index(f)] = 1

    # длина зоны одного типа и позиция внутри неё
    zone_len = np.zeros(n); zone_pos = np.zeros(n); zone_idx = np.zeros(n)
    if have_marks:
        cur = None; i = 0; cnt = Counter()
        while i < n:
            t = TYPES[int(np.argmax(TY[i]))] if TY[i].any() else "?"
            j = i
            while j < n and (TYPES[int(np.argmax(TY[j]))] if TY[j].any() else "?") == t:
                j += 1
            L = dur[i:j].sum()
            cnt[t] += 1
            for k in range(i, j):
                zone_len[k] = L
                zone_pos[k] = (mid[k] - start[i]) / max(L, 1e-6)
                zone_idx[k] = cnt[t]
            i = j

    # ---- новизна: похожесть слов блока на всё, что было раньше
    seen = Counter()
    novelty = np.zeros(n)
    for i in range(n):
        w = set(re.findall(r"\w{4,}", text[i].lower()))
        if w:
            novelty[i] = sum(1 for x in w if seen[x] == 0) / len(w)
        for x in w:
            seen[x] += 1

    # ---- окрестности
    ctx = []
    for sec in (30, 120, 600):
        ctx.append(roll(dens_char * is_sp, mid, dur, sec))
        ctx.append(roll(is_sp, mid, dur, sec))
        ctx.append(roll(vol_rel, mid, dur, sec))
        ctx.append(roll(novelty, mid, dur, sec))
        if have_marks:
            for j in range(len(TYPES)):
                ctx.append(roll(TY[:, j], mid, dur, sec))

    X = np.column_stack([
        dur, nchar, nword, dens_char, dens_word, wlen, is_sp,
        pause_before, pause_after, vol_rel, novelty,
        mid / T, np.minimum(mid / 60, 40), np.minimum((T - mid) / 60, 40),
        zone_len, zone_pos, zone_idx,
        clap, TY, FL, np.column_stack(ctx),
    ]).astype(np.float32)
    y = np.array([b["used_frac"] for b in B], np.float32)
    os.makedirs(f"{DATA}/feat", exist_ok=True)
    np.savez(f"{DATA}/feat/s{s}.npz", X=X, y=y, dur=dur, start=start,
             kind=is_sp, marks=have_marks)
    print(f"слот {s}: {X.shape[0]} блоков, {X.shape[1]} признаков, "
          f"типы {'есть' if have_marks else 'НЕТ'}")


if __name__ == "__main__":
    main()
