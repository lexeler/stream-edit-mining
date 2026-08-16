#!/usr/bin/env python3
"""
Размечает ленту стрима: попал элемент в итоговый выпуск или нет.

Источник истины — монтажный лист из проекта Premiere (410 отрезков в
координатах стрима). Пересёкся хоть на сколько-то -> used = true.
"""
import bisect, json, os

OUT = os.environ.get("SEM_DATASET", "work/dataset")
EDL = os.path.join(os.environ.get("SEM_STREAMS", "work/streams"), "7/narezki/timecodes.json")
SRC = f"{OUT}/stream_2026-06-30_varenikovskiy.json"
DST = f"{OUT}/stream_2026-06-30_varenikovskiy.labeled.json"


def main():
    edl = json.load(open(EDL))["segments"]
    segs = sorted((s["start"], s["end"]) for s in edl)
    starts = [a for a, _ in segs]
    doc = json.load(open(SRC))
    items = doc["items"]

    def overlap(a, b):
        """сколько секунд [a,b] пересекается с монтажным листом"""
        i = max(0, bisect.bisect_right(starts, a) - 1)
        tot = 0.0
        while i < len(segs) and segs[i][0] < b:
            tot += max(0.0, min(b, segs[i][1]) - max(a, segs[i][0]))
            i += 1
        return tot

    for it in items:
        d = it["end"] - it["start"]
        ov = overlap(it["start"], it["end"])
        it["used"] = ov > 0
        it["used_frac"] = round(ov / d, 3) if d > 0 else 0.0

    w = [i for i in items if i["type"] == "word"]
    s = [i for i in items if i["type"] == "silence"]
    wk = sum(1 for i in w if i["used"])
    sk = sum(1 for i in s if i["used"])
    kept_t = sum(i["end"] - i["start"] for i in items if i["used"])
    part = sum(1 for i in items if 0 < i["used_frac"] < 1)

    doc["labels"] = {
        "source_edl": os.path.basename(EDL),
        "edl_segments": len(segs),
        "edl_duration_sec": round(sum(b - a for a, b in segs), 2),
        "rule": "used=true если элемент пересекается с монтажным листом хотя бы частично",
        "words_used": wk, "words_dropped": len(w) - wk,
        "silences_used": sk, "silences_dropped": len(s) - sk,
        "time_used_sec": round(kept_t, 2),
        "partial_items": part,
    }
    json.dump(doc, open(DST, "w"), ensure_ascii=False, indent=1)

    print(f"монтажный лист: {len(segs)} отрезков, "
          f"{sum(b-a for a,b in segs)/60:.1f} мин")
    print(f"слов:  оставлено {wk} из {len(w)}  ({wk/len(w)*100:.1f}%)")
    print(f"пауз:  оставлено {sk} из {len(s)}  ({sk/len(s)*100:.1f}%)")
    print(f"времени оставлено: {kept_t/60:.1f} из {doc['duration_sec']/60:.1f} мин")
    print(f"частично задетых элементов: {part}")
    print(f"-> {os.path.basename(DST)} ({os.path.getsize(DST)/2**20:.1f} МБ)")


if __name__ == "__main__":
    main()
