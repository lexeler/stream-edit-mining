#!/usr/bin/env python3
"""Проверяем кандидатов в признаки: сколько информации каждый даёт о метке."""
import os
import json, math, re
import numpy as np

OUT = os.environ.get("SEM_DATASET", "work/dataset")
d = json.load(open(f"{OUT}/stream_2026-06-30_varenikovskiy.labeled.json"))
I = d["items"]
N = len(I)
DUR = d["duration_sec"]
y = np.array([1 if i["used"] else 0 for i in I])

FILLER = {"ну", "вот", "это", "как", "бы", "типа", "короче", "э", "а", "блин",
          "значит", "самое", "такой", "там"}
MAT = re.compile(r"(бля|хуй|хуе|хуё|пизд|ебан|ёбан|еба|ёба|сук[аи]|нахуй|пидор|говн)", re.I)


def mi(feat, yv, bins=8):
    """взаимная информация признака с меткой, в битах"""
    f = np.asarray(feat, dtype=float)
    if len(set(f.round(6))) < 3:
        b = f.astype(int)
    else:
        qs = np.unique(np.quantile(f, np.linspace(0, 1, bins + 1)))
        b = np.clip(np.digitize(f, qs[1:-1]), 0, len(qs) - 2)
    tot = 0.0
    py = np.array([1 - yv.mean(), yv.mean()])
    for v in np.unique(b):
        m = b == v
        px = m.mean()
        if px == 0:
            continue
        p1 = yv[m].mean()
        for c, pc in ((1, p1), (0, 1 - p1)):
            if pc > 0:
                tot += px * pc * math.log2(pc / py[c])
    return tot


def spread(feat, yv, bins=6):
    """разброс доли used между корзинами — насколько наглядно разделяет"""
    f = np.asarray(feat, dtype=float)
    qs = np.unique(np.quantile(f, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(f, qs[1:-1]), 0, len(qs) - 2)
    r = [yv[b == v].mean() for v in np.unique(b) if (b == v).sum() > 50]
    return (max(r) - min(r)) * 100 if r else 0.0


# ---- собираем кандидатов -------------------------------------------------
dur = np.array([i["end"] - i["start"] for i in I])
isw = np.array([1 if i["type"] == "word" else 0 for i in I])
vol = np.array([i.get("vol", 0) for i in I])
pos = np.array([i["start"] / DUR for i in I])

pause_before = np.zeros(N)
pause_after = np.zeros(N)
for k in range(N):
    if k and I[k - 1]["type"] == "silence":
        pause_before[k] = I[k - 1]["end"] - I[k - 1]["start"]
    if k + 1 < N and I[k + 1]["type"] == "silence":
        pause_after[k] = I[k + 1]["end"] - I[k + 1]["start"]

# плотность речи в окне +-15 с
starts = np.array([i["start"] for i in I])
speech = np.where(isw == 1, dur, 0.0)
cs = np.concatenate(([0.0], np.cumsum(speech)))
cd = np.concatenate(([0.0], np.cumsum(dur)))
lo = np.searchsorted(starts, starts - 15)
hi = np.searchsorted(starts, starts + 15)
dens = (cs[hi] - cs[lo]) / np.maximum(cd[hi] - cd[lo], 1e-6)

# громкость относительно окружения
vcs = np.concatenate(([0.0], np.cumsum(vol)))
vloc = (vcs[hi] - vcs[lo]) / np.maximum(hi - lo, 1)
vrel = vol - vloc

wtxt = [i.get("word", "").strip(".,!?»«:;()-—").lower() for i in I]
filler = np.array([1 if w in FILLER else 0 for w in wtxt])
mat = np.array([1 if MAT.search(w) else 0 for w in wtxt])
wlen = np.array([len(w) for w in wtxt])
rep = np.array([1 if k and wtxt[k] and wtxt[k] == wtxt[k - 2 if k > 1 else 0] else 0
                for k in range(N)])
endsent = np.array([1 if i.get("word", "").endswith((".", "!", "?")) else 0 for i in I])

CAND = [
    ("позиция в стриме", pos),
    ("длина паузы перед", pause_before),
    ("длина паузы после", pause_after),
    ("плотность речи вокруг", dens),
    ("длительность элемента", dur),
    ("громкость абсолютная", vol),
    ("громкость отн. окружения", vrel),
    ("слово или тишина", isw),
    ("длина слова в буквах", wlen),
    ("слово-паразит", filler),
    ("мат", mat),
    ("конец предложения", endsent),
    ("повтор слова", rep),
]

print(f"{'признак':<28}{'инфо, бит':>11}{'разброс used, п.п.':>21}")
print("-" * 62)
for name, f in sorted(CAND, key=lambda c: -mi(c[1], y)):
    print(f"{name:<28}{mi(f, y):>10.4f}{spread(f, y):>19.1f}")

print(f"\nэнтропия метки: {-(y.mean()*math.log2(y.mean())+(1-y.mean())*math.log2(1-y.mean())):.4f} бит")
print("(инфо 0.01 бита = признак объясняет ~1.5% неопределённости)")

print("\nдетали по слабым, но любопытным:")
for name, f in (("слово-паразит", filler), ("мат", mat), ("конец предложения", endsent)):
    for v in (0, 1):
        m = f == v
        if m.sum() > 50:
            print(f"  {name}={v}: {m.sum():>6} шт, оставлено {y[m].mean()*100:5.1f}%")
