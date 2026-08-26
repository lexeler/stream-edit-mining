#!/usr/bin/env python3
"""
Читаемое представление стрима для языковой модели.

Лента элементов «слово — тишина» превращается в реплики: слова склеиваются,
пока пауза между ними меньше GAP. Реплика — единица решения, её медианная
длина близка к медианному куску монтажёра (4–9 с), поэтому потолок такой
разбивки выше, чем у сетки в 10 секунд.

На выходе два файла:
  <имя>.blocks.json   блоки с таймкодами и меткой used_frac (если есть)
  <имя>.txt           текст для подачи модели

  view.py Обучение/slot5_2026-06-12_matiz.labeled.json out/s5
"""
import json, sys, os, argparse
import numpy as np

GAP = 0.7          # пауза, разрывающая реплику, с
LONG_PAUSE = 3.0   # паузу длиннее показываем отдельной строкой
MAXLEN = 25.0      # реплика длиннее — режем принудительно


def build_blocks(items):
    """items -> список блоков {i, start, end, text, sil_before, used_frac}"""
    blocks = []
    cur = None
    pending_sil = 0.0
    for r in items:
        if r["type"] == "silence":
            d = r["end"] - r["start"]
            if cur is not None and d >= GAP:
                cur["end"] = r["start"]
                blocks.append(cur); cur = None
                pending_sil = d
            elif cur is not None:
                pending_sil = 0.0
            else:
                pending_sil += d
            continue
        if cur is not None and r["end"] - cur["start"] > MAXLEN:
            cur["end"] = r["start"]
            blocks.append(cur); cur = None
            pending_sil = 0.0
        if cur is None:
            cur = {"start": r["start"], "end": r["end"], "words": [],
                   "sil_before": round(pending_sil, 2)}
            pending_sil = 0.0
        cur["words"].append(r["word"])
        cur["end"] = r["end"]
    if cur is not None:
        blocks.append(cur)
    for i, b in enumerate(blocks):
        b["i"] = i
        b["text"] = " ".join(b["words"]); del b["words"]
        b["start"] = round(b["start"], 2); b["end"] = round(b["end"], 2)
    return blocks


def label_blocks(blocks, items):
    """доля блока, вошедшая в выпуск, по разметке элементов"""
    if "used_frac" not in items[0]:
        return
    st = np.array([r["start"] for r in items], np.float64)
    en = np.array([r["end"] for r in items], np.float64)
    fr = np.array([r.get("used_frac", 0.0) for r in items], np.float64)
    for b in blocks:
        lo = np.maximum(st, b["start"]); hi = np.minimum(en, b["end"])
        ov = np.maximum(hi - lo, 0)
        d = ov.sum()
        b["used_frac"] = round(float((ov * fr).sum() / d), 3) if d > 0 else 0.0


def mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def render(blocks, labeled, extras=None):
    """extras: dict i -> строка с дополнительными пометками (кадр, звук)"""
    out = []
    for b in blocks:
        if b["sil_before"] >= LONG_PAUSE:
            out.append(f"    ... пауза {b['sil_before']:.0f} с ...")
        mark = ""
        if labeled:
            f = b.get("used_frac", 0.0)
            mark = "  ОСТАВИЛ" if f > 0.6 else ("  выбросил" if f < 0.25
                                                else f"  частично {f:.0%}")
        ex = ""
        if extras and b["i"] in extras:
            ex = "\n      " + extras[b["i"]]
        out.append(f"[{b['i']:04d}] {mmss(b['start'])} ({b['end']-b['start']:.0f}с) "
                   f"{b['text']}{mark}{ex}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    d = json.load(open(a.src))
    items = d["items"]
    labeled = "used_frac" in items[0]
    blocks = build_blocks(items)
    if labeled:
        label_blocks(blocks, items)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"source": d.get("source"), "duration_sec": d.get("duration_sec"),
               "gap": GAP, "blocks": blocks},
              open(a.out + ".blocks.json", "w"), ensure_ascii=False)
    txt = render(blocks, labeled)
    open(a.out + ".txt", "w").write(txt)
    ln = np.array([b["end"] - b["start"] for b in blocks])
    cov = ln.sum() / d["duration_sec"]
    print(f"{os.path.basename(a.src)}: {len(blocks)} блоков, "
          f"медиана {np.median(ln):.1f} с, речь покрывает {cov:.0%} эфира, "
          f"текст {len(txt)/1000:.0f} тыс. символов (~{len(txt)//3000} тыс. токенов)")
    if labeled:
        f = np.array([b.get("used_frac", 0) for b in blocks])
        print(f"  оставлено блоков {(f>0.6).mean():.1%}, частично "
              f"{((f>=0.25)&(f<=0.6)).mean():.1%}, выброшено {(f<0.25).mean():.1%}")


if __name__ == "__main__":
    main()
