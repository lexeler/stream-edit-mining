#!/usr/bin/env python3
"""
Подготовка стрима к работе языковой модели.

Лента элементов -> сплошная лента блоков-реплик. Блок начинается там, где
кончился предыдущий (то есть пауза перед репликой принадлежит ей), длинные
паузы выделяются отдельно. Потолок такой разбивки замерен в oracle.py:
точность 0,871 при GAP 1,2 — выше, чем у сетки в 10 секунд (0,827).

  prep.py --slot 5            обучающий слот, с ответами
  prep.py --slot 2 --test     тестовый, ответы в отдельный файл

Кладёт в data/:
  s<N>.blocks.json   блоки с таймкодами, текстом и меткой (если есть)
  s<N>.txt           представление для модели БЕЗ меток
  s<N>.marked.txt    то же С метками — для ручного разбора правил
"""
import json, os, argparse, glob
import numpy as np

DATASET = os.environ.get("SEM_DATASET", "work/dataset")
OUT = os.environ.get("SEM_BLOCKS", "work/blocks")
# Пауза от 0,8 с разрывает реплику И становится отдельным блоком.
# Замер по семи выпускам: медиана ВЫБРОШЕННОГО монтажёром промежутка —
# 1,0–1,8 с, то есть он режет именно короткие паузы. При склейке пауз
# короче 3 с их нельзя выбросить поштучно, и нарезка выходит монолитной.
# Потолок точности при 0,8 с: 0,924 против 0,871 при 1,2 с.
GAP = 0.8
LONG = 0.8

SRC = {
    "1": "slot1_2026-07-30_futbol-mozgi.labeled.json",
    "2": "slot2_2026-07-31_gulyanka.json",
    "3": "slot3_2026-08-03_tarzanka-zabroshka.labeled.json",
    "4": "slot4_2026-08-04_razvlekalovka.labeled.json",
    "5": "slot5_2026-06-12_matiz.labeled.json",
    "6": "slot6_2026-06-30_2mp4.labeled.json",
    "7": "slot7_2026-06-30_varenikovskiy.labeled.json",
}
ANS2 = "slot2_2026-07-31_gulyanka.answers.json"


def build(items, gap=GAP, long_pause=LONG):
    out, cur, seg = [], None, 0.0
    for r in items:
        if r["type"] == "silence":
            d = r["end"] - r["start"]
            if cur is not None and d >= gap:
                out.append({"start": seg, "end": r["start"],
                            "text": " ".join(cur), "kind": "speech"})
                cur, seg = None, r["start"]
            if cur is None and d >= long_pause:
                if r["start"] > seg + 1e-9:
                    out.append({"start": seg, "end": r["start"], "text": "",
                                "kind": "pause"})
                out.append({"start": r["start"], "end": r["end"], "text": "",
                            "kind": "pause"})
                seg = r["end"]
            continue
        if cur is None:
            cur = []
        cur.append(r["word"])
    last = items[-1]["end"]
    if cur is not None:
        out.append({"start": seg, "end": last, "text": " ".join(cur),
                    "kind": "speech"})
    elif last > seg + 1e-9:
        out.append({"start": seg, "end": last, "text": "", "kind": "pause"})
    # исходная лента режет длинные паузы на куски по 1–5 с (по минимумам
    # громкости), поэтому подряд идущие «тишины» надо слить обратно: иначе
    # в представлении десять строк «тишина 4 с» вместо одной «тишина 40 с»
    merged = []
    for b in out:
        if (merged and b["kind"] == "pause" and merged[-1]["kind"] == "pause"
                and abs(merged[-1]["end"] - b["start"]) < 1e-6):
            merged[-1]["end"] = b["end"]
        else:
            merged.append(b)
    for i, b in enumerate(merged):
        b["i"] = i
        b["start"] = round(b["start"], 2)
        b["end"] = round(b["end"], 2)
        b["dur"] = round(b["end"] - b["start"], 2)
    return merged


def label(blocks, items, key="used_frac"):
    st = np.array([r["start"] for r in items], np.float64)
    en = np.array([r["end"] for r in items], np.float64)
    fr = np.array([r.get(key, 0.0) or 0.0 for r in items], np.float64)
    for b in blocks:
        lo = np.maximum(st, b["start"]); hi = np.minimum(en, b["end"])
        ov = np.maximum(hi - lo, 0.0)
        d = ov.sum()
        b["used_frac"] = round(float((ov * fr).sum() / d), 3) if d > 0 else 0.0


def mmss(t):
    return f"{int(t)//60:>3}:{int(t)%60:02d}"


def render(blocks, marked):
    lines = []
    for b in blocks:
        if b["kind"] == "pause":
            if b["dur"] >= LONG:
                lines.append(f"[{b['i']}] {mmss(b['start'])} — тишина {b['dur']:.0f} с"
                             + (mark(b) if marked else ""))
            continue
        lines.append(f"[{b['i']}] {mmss(b['start'])} ({b['dur']:.0f}с) {b['text']}"
                     + (mark(b) if marked else ""))
    return "\n".join(lines)


def mark(b):
    f = b.get("used_frac", 0.0)
    if f > 0.6:
        return "   <<< ОСТАВЛЕНО"
    if f < 0.25:
        return "   <<< выброшено"
    return f"   <<< частично {f:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    a = ap.parse_args()
    s = a.slot
    src = f"{DATASET}/{SRC[s]}"
    d = json.load(open(src))
    items = d["items"]
    if s == "2":
        ans = json.load(open(f"{DATASET}/{ANS2}"))["answers"]
        for r, q in zip(items, ans):
            r["used_frac"] = q["used_frac"]
    blocks = build(items)
    label(blocks, items)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"slot": s, "source": d.get("source"),
               "duration_sec": d.get("duration_sec"), "gap": GAP,
               "blocks": blocks}, open(f"{OUT}/s{s}.blocks.json", "w"),
              ensure_ascii=False)
    open(f"{OUT}/s{s}.txt", "w").write(render(blocks, False))
    open(f"{OUT}/s{s}.marked.txt", "w").write(render(blocks, True))
    dur = np.array([b["dur"] for b in blocks])
    fr = np.array([b["used_frac"] for b in blocks])
    sp = np.array([b["kind"] == "speech" for b in blocks])
    keep = fr > 0.5
    ov = (dur * fr)[keep].sum()
    n = len(open(f"{OUT}/s{s}.txt").read())
    print(f"слот {s}: {len(blocks)} блоков ({sp.sum()} речь, {(~sp).sum()} тишина), "
          f"эфир {dur.sum()/60:.0f} мин, выпуск {(dur*fr).sum()/60:.0f} мин "
          f"({(dur*fr).sum()/dur.sum():.1%})")
    print(f"  потолок разбивки: точность {ov/(dur[keep].sum()):.3f}, "
          f"полнота {ov/(dur*fr).sum():.3f}")
    print(f"  текст {n/1000:.0f} тыс. символов (~{n//3000} тыс. токенов)")


if __name__ == "__main__":
    main()
