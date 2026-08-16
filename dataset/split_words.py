#!/usr/bin/env python3
"""
Разрезает слова длиннее 5 секунд: тишина внутри становится отдельным блоком,
звучащая часть остаётся словом.

Такие «слова» — артефакт модели: она приписывает последнему распознанному
слову весь хвост до следующей реплики. У «устал.» 7.32 с из них 85% тишины.
"""
import bisect, json, os
import numpy as np

OUT = os.environ.get("SEM_DATASET", "work/dataset")
RAW = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "edl/s7.raw")
EDL = os.path.join(os.environ.get("SEM_STREAMS", "work/streams"), "7/narezki/timecodes.json")
NAME = "stream_2026-06-30_varenikovskiy"

SR = 8000
LONG = 5.0        # порог «подозрительно длинного» слова
WIN = 0.025       # окно замера уровня
MIN_SIL = 0.30    # тишина короче не разрывает слово (это дыхание, а не пауза)
REL = 0.08        # порог тишины: доля от пика внутри слова
ABS = 0.005       # абсолютный пол


def silent_runs(seg):
    """границы тишины внутри куска, в секундах от его начала"""
    w = int(WIN * SR)
    n = len(seg) // w
    if n < 2:
        return []
    rms = np.sqrt((seg[:n * w].reshape(n, w) ** 2).mean(axis=1))
    thr = max(rms.max() * REL, ABS)
    quiet = rms < thr
    runs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * WIN >= MIN_SIL:
                runs.append((i * WIN, j * WIN))
            i = j
        else:
            i += 1
    return runs


def main():
    x = np.fromfile(RAW, dtype="<i2").astype(np.float32) / 32768
    doc = json.load(open(f"{OUT}/{NAME}.json"))
    items = doc["items"]

    new, report = [], []
    for it in items:
        dur = it["end"] - it["start"]
        if it["type"] != "word" or dur <= LONG:
            new.append(it)
            continue
        seg = x[int(it["start"] * SR):int(it["end"] * SR)]
        runs = silent_runs(seg)
        if not runs:
            new.append(it)
            report.append((it["word"], dur, dur, 0))
            continue
        # слово занимает всё до первой длинной тишины
        first = runs[0][0]
        if first < 0.08:          # тишина в самом начале — слово где-то дальше
            first = max(0.08, runs[0][1])
        word_end = it["start"] + first
        new.append({"type": "word", "word": it["word"],
                    "start": it["start"], "end": round(word_end, 2)})
        new.append({"type": "silence", "start": round(word_end, 2),
                    "end": it["end"]})
        report.append((it["word"], dur, first, it["end"] - word_end))

    # сливаем подряд идущие тишины
    merged = []
    for it in new:
        if (merged and it["type"] == "silence" and merged[-1]["type"] == "silence"
                and abs(merged[-1]["end"] - it["start"]) < 1e-6):
            merged[-1]["end"] = it["end"]
        else:
            merged.append(dict(it))

    print(f"{'слово':<14}{'было':>8}{'стало словом':>14}{'ушло в тишину':>15}")
    print("-" * 52)
    for w, d0, d1, d2 in report:
        print(f"{w[:13]:<14}{d0:7.2f}с{d1:13.2f}с{d2:14.2f}с")

    nw = sum(1 for i in merged if i["type"] == "word")
    ns = len(merged) - nw
    freed = sum(r[3] for r in report)
    print(f"\nэлементов: {len(items)} -> {len(merged)}  (слов {nw}, тишины {ns})")
    print(f"переведено из слов в тишину: {freed:.1f} с")

    doc["items"] = merged
    doc["words_total"] = nw
    doc["silences_total"] = ns
    doc["postprocess"] = (f"слова длиннее {LONG} с разрезаны: тишина внутри "
                          f"(от {MIN_SIL} с) вынесена отдельным блоком")
    json.dump(doc, open(f"{OUT}/{NAME}.json", "w"), ensure_ascii=False, indent=1)

    # заново размечаем по монтажному листу
    segs = sorted((s["start"], s["end"])
                  for s in json.load(open(EDL))["segments"])
    starts = [a for a, _ in segs]

    def cut(a, b):
        cuts = {a, b}
        i = max(0, bisect.bisect_right(starts, a) - 1)
        while i < len(segs) and segs[i][0] < b:
            for v in segs[i]:
                if a < v < b:
                    cuts.add(v)
            i += 1
        xs = sorted(cuts)
        out = []
        for s, e in zip(xs, xs[1:]):
            m = (s + e) / 2
            j = max(0, bisect.bisect_right(starts, m) - 1)
            out.append((s, e, any(segs[k][0] <= m < segs[k][1]
                                  for k in range(j, min(j + 2, len(segs))))))
        return out

    def hit(a, b):
        i = max(0, bisect.bisect_right(starts, a) - 1)
        while i < len(segs) and segs[i][0] < b:
            if min(b, segs[i][1]) > max(a, segs[i][0]):
                return True
            i += 1
        return False

    lab = []
    for it in merged:
        if it["type"] == "word":
            lab.append({**it, "used": hit(it["start"], it["end"])})
        else:
            for s, e, u in cut(it["start"], it["end"]):
                if e - s >= 0.02:
                    lab.append({"type": "silence", "start": round(s, 2),
                                "end": round(e, 2), "used": u})
    W = [i for i in lab if i["type"] == "word"]
    S = [i for i in lab if i["type"] == "silence"]
    wu = sum(1 for i in W if i["used"])
    su = sum(1 for i in S if i["used"])
    tu = sum(i["end"] - i["start"] for i in lab if i["used"])
    doc2 = dict(doc)
    doc2["items"] = lab
    doc2["labels"] = {
        "source_edl": "7/narezki/timecodes.json",
        "edl_segments": len(segs),
        "edl_duration_sec": round(sum(b - a for a, b in segs), 2),
        "rule": "слова атомарны; тишина разрезана по границам монтажа",
        "words_used": wu, "words_dropped": len(W) - wu,
        "silences_used": su, "silences_dropped": len(S) - su,
        "time_used_sec": round(tu, 2),
    }
    json.dump(doc2, open(f"{OUT}/{NAME}.labeled.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nразметка: слов used {wu}/{len(W)} ({wu/len(W)*100:.1f}%), "
          f"тишины used {su}/{len(S)} ({su/len(S)*100:.1f}%)")
    print(f"времени used: {tu/60:.1f} мин (монтажный лист 64.4)")


if __name__ == "__main__":
    main()
