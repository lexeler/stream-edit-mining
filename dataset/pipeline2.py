#!/usr/bin/env python3
"""
Второй проход: убираем переоценку и даём модели предсказуемую цель.

Два изменения против первой версии.

1. Паузы режутся не по механической сетке в 5 секунд, а по МИНИМУМАМ ГРОМКОСТИ.
   Замерено: монтажёр в 63% случаев режет в самой тихой четверти окрестности,
   и почти никогда (2%) в громкой. Значит границы наших элементов надо ставить
   туда же — тогда элемент чаще оказывается целиком внутри монтажа или целиком
   снаружи, и приписывать нечего.

2. Метка перестаёт быть булевой. Основная цель — used_frac, доля элемента,
   вошедшая в выпуск. Плюс где именно проходит рез внутри элемента.

Утечки нет: границы элементов считаются только по звуку, монтажный лист
участвует лишь на этапе навешивания меток.
"""
import bisect, json, math, os, re, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (SLOTS, BASE, OUT, TMP, SR8, LONG_W, MIN_GAP, REL, ABS_,
                      DB_LO, DB_HI, FILLER, MAT, sh, log, silent_runs,
                      frame_feats, add_features)

SIL_MIN, SIL_MAX = 1.0, 5.0     # длина куска тишины: от и до


def split_by_minima(a, b, rms):
    """Режем тишину [a,b] по самым тихим точкам, куски от SIL_MIN до SIL_MAX."""
    if b - a <= SIL_MAX:
        return [(a, b)]
    out, t = [], a
    while b - t > SIL_MAX:
        lo, hi = int((t + SIL_MIN) * 100), int(min(t + SIL_MAX, b - SIL_MIN) * 100)
        if hi <= lo:
            cut = t + SIL_MAX
        else:
            w = rms[lo:hi]
            cut = (lo + int(np.argmin(w))) / 100.0 if len(w) else t + SIL_MAX
        cut = round(max(t + SIL_MIN, min(cut, b - SIL_MIN)), 2)
        out.append((t, cut))
        t = cut
    out.append((t, b))
    return out


def build_items(words, dur, x, rms):
    items, prev = [], 0.0
    for s, e, t in words:
        if s - prev >= 0.04:
            items.append({"type": "silence", "start": round(prev, 2), "end": round(s, 2)})
        items.append({"type": "word", "word": t, "start": round(s, 2), "end": round(e, 2)})
        prev = max(prev, e)
    if dur - prev >= 0.04:
        items.append({"type": "silence", "start": round(prev, 2), "end": round(dur, 2)})

    out = []
    for it in items:                      # длинные слова: тишина внутри -> блок
        d = it["end"] - it["start"]
        if it["type"] != "word" or d <= LONG_W:
            out.append(it); continue
        runs = silent_runs(x[int(it["start"] * SR8):int(it["end"] * SR8)])
        if not runs:
            out.append(it); continue
        first = runs[0][0] if runs[0][0] >= 0.08 else max(0.08, runs[0][1])
        we = round(it["start"] + first, 2)
        out.append({"type": "word", "word": it["word"], "start": it["start"], "end": we})
        out.append({"type": "silence", "start": we, "end": it["end"]})

    m = []
    for it in out:                        # слить соседние тишины
        if (m and it["type"] == "silence" and m[-1]["type"] == "silence"
                and abs(m[-1]["end"] - it["start"]) < 1e-6):
            m[-1]["end"] = it["end"]
        else:
            m.append(dict(it))

    fin = []
    for it in m:                          # длинные тишины -> по минимумам громкости
        if it["type"] != "silence":
            fin.append(it); continue
        for a, b in split_by_minima(it["start"], it["end"], rms):
            if b - a >= 0.02:
                fin.append({"type": "silence", "start": round(a, 2), "end": round(b, 2)})

    # лента должна быть сплошной: конец элемента = начало следующего.
    # иначе окно "5 минут назад / вперёд" будет рваным, а нахлёсты дадут
    # один и тот же звук в двух элементах с разными метками.
    fin.sort(key=lambda i: (i["start"], i["end"]))
    snapped = []
    for it in fin:
        if snapped:
            p = snapped[-1]
            if it["start"] < p["end"]:
                if it["start"] <= p["start"] + 0.02:
                    continue                      # элемент утонул в предыдущем
                p["end"] = round(it["start"], 2)
            elif it["start"] > p["end"]:
                p["end"] = round(it["start"], 2)  # дыра -> отдаём предыдущему
        if it["end"] - it["start"] >= 0.02:
            snapped.append(it)
    if snapped:
        snapped[-1]["end"] = round(max(snapped[-1]["end"], dur), 2)
    return snapped


def label(items, edl_path):
    segs = sorted((s["start"], s["end"])
                  for s in json.load(open(edl_path))["segments"])
    st = [a for a, _ in segs]
    out = []
    for it in items:
        a, b = it["start"], it["end"]
        d = b - a
        i = max(0, bisect.bisect_right(st, a) - 1)
        ov, first, last = 0.0, None, None
        while i < len(segs) and segs[i][0] < b:
            lo, hi = max(a, segs[i][0]), min(b, segs[i][1])
            if hi > lo:
                ov += hi - lo
                if first is None:
                    first = lo
                last = hi
            i += 1
        frac = ov / d if d > 0 else 0.0
        rec = {"used": ov > 0, "used_half": bool(frac >= 0.5),
               "used_frac": round(frac, 3), "cut_rel": None, "cut_side": None}
        if 0 < frac < 1 and first is not None:
            head = abs(first - a) < 1e-6          # взятая часть начинается с начала
            tail = abs(last - b) < 1e-6           # и/или доходит до конца
            if head and not tail:
                rec["cut_side"] = "keep_head"
                rec["cut_rel"] = round((last - a) / d, 3)
            elif tail and not head:
                rec["cut_side"] = "keep_tail"
                rec["cut_rel"] = round((first - a) / d, 3)
            else:
                rec["cut_side"] = "keep_middle"
                rec["cut_rel"] = round((first - a) / d, 3)
        out.append(rec)
    return out, len(segs), sum(b - a for a, b in segs)


def process(slot, name, split):
    video = f"{BASE}/{slot}/streams/full_video.mp4"
    edl = f"{BASE}/{slot}/narezki/timecodes.json"
    cache = f"{TMP}/{name}.words.json"
    if not os.path.exists(cache):
        log(f"  СЛОТ {slot}: нет кэша распознавания, пропускаю")
        return None
    log(f"\n{'='*66}\nСЛОТ {slot} -> {name}  [{split}]\n{'='*66}")
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", video]).stdout.strip())
    raw = f"{TMP}/{name}.raw"
    if not os.path.exists(raw):
        sh(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1",
            "-ar", str(SR8), "-f", "s16le", "-y", raw])
    x = np.fromfile(raw, dtype="<i2").astype(np.float32) / 32768
    W = int(0.01 * SR8)
    n = len(x) // W
    rms = np.sqrt((x[:n * W].reshape(n, W) ** 2).mean(axis=1))

    words = json.load(open(cache))
    items = build_items(words, dur, x, rms)
    nw = sum(1 for i in items if i["type"] == "word")
    sils = [i["end"] - i["start"] for i in items if i["type"] == "silence"]
    log(f"  элементов {len(items)} ({nw} слов, {len(sils)} тишины), "
        f"макс. пауза {max(sils):.2f} с")

    items = add_features(items, x, dur)
    del x
    lab, nseg, edur = label(items, edl)
    used = sum(1 for l in lab if l["used"])
    t_used = sum(i["end"] - i["start"] for i, l in zip(items, lab) if l["used"])
    t_frac = sum((i["end"] - i["start"]) * l["used_frac"] for i, l in zip(items, lab))
    part = sum(1 for l in lab if 0 < l["used_frac"] < 1)
    log(f"  монтаж {nseg} отрезков, {edur/60:.1f} мин")
    log(f"  used {used}/{len(items)} = {used/len(items)*100:.1f}%, "
        f"время {t_used/60:.1f} мин (переоценка {(t_used-edur)/60:+.1f})")
    log(f"  по used_frac: {t_frac/60:.1f} мин — расхождение {(t_frac-edur):+.0f} с")
    log(f"  пограничных элементов: {part} ({part/len(items)*100:.1f}%)")

    head = {"slot": slot, "split": split, "source": os.path.basename(video),
            "duration_sec": round(dur, 3), "model": "GigaAM v3 e2e_rnnt",
            "items_total": len(items), "words_total": nw,
            "silences_total": len(items) - nw,
            "pipeline": ("слова >5с разрезаны по внутренней тишине; паузы >5с "
                         "разрезаны по минимумам громкости на куски 1-5 с; "
                         "границы элементов зависят только от звука"),
            "targets": {
                "used_frac": "основная цель: доля элемента, вошедшая в выпуск (0..1)",
                "used": "задето хотя бы частично",
                "used_half": "вошло больше половины",
                "cut_rel": "где внутри элемента проходит граница монтажа (0..1)",
                "cut_side": "keep_head — остаётся начало, keep_tail — конец, "
                            "keep_middle — середина",
            },
            "features": list(items[0]["f"].keys())}

    json.dump({**head, "items": items}, open(f"{OUT}/{name}.json", "w"),
              ensure_ascii=False)
    body = {**head, "edl_segments": nseg, "edl_duration_sec": round(edur, 2)}
    if split == "train":
        json.dump({**body, "items": [{**i, **l} for i, l in zip(items, lab)]},
                  open(f"{OUT}/{name}.labeled.json", "w"), ensure_ascii=False)
    else:
        json.dump({**body, "answers": [{"i": k, "start": i["start"], "end": i["end"], **l}
                                       for k, (i, l) in enumerate(zip(items, lab))]},
                  open(f"{OUT}/{name}.answers.json", "w"), ensure_ascii=False)
    if os.path.exists(raw):
        os.remove(raw)
    return {"slot": slot, "name": name, "split": split, "items": len(items),
            "used": used, "over": (t_used - edur) / 60, "part": part}


if __name__ == "__main__":
    res = []
    for s in sys.argv[1:] or list(SLOTS):
        n, sp = SLOTS[s]
        r = process(s, n, sp)
        if r:
            res.append(r)
    log(f"\n{'='*66}")
    log(f"{'слот':>5} {'элементов':>10} {'used':>7} {'переоценка':>12} {'погранич.':>10}")
    for r in res:
        log(f"{r['slot']:>5} {r['items']:>10} {r['used']/r['items']*100:>6.1f}% "
            f"{r['over']:>+11.1f}м {r['part']:>10}")
