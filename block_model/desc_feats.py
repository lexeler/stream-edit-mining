#!/usr/bin/env python3
"""
Признаки из описаний кадров: где герои и что делают.

Разбор слота 4 показал главное: решение определяется тем, ГДЕ мы находимся,
а не тем, что сказано. Один и тот же возглас внутри аттракциона остаётся
в 67 % случаев, а на улице между делом — в 4 %. В расшифровке этого нет.

Видео-языковая модель описывает кадр раз в 12 секунд одной строкой
(«улица, двое мужчин, один режет арбуз»). Отсюда берём:
  мешок частых слов — сама обстановка;
  сколько людей в кадре;
  внутри / на улице / в машине;
  новизна описания — было ли такое место раньше в этом же стриме;
  длина текущей однородной зоны и номер её посещения.

Проверка на слоте 7: «старик» +41 процентный пункт к доле оставленного,
«кафе/едят/столом» −21, «помещении» −9. То есть сигнал есть.
"""
import json, os, re, math
from collections import Counter
import numpy as np

SEC = 12.0          # шаг кадров
NUMWORDS = {"один": 1, "одна": 1, "двое": 2, "два": 2, "две": 2, "трое": 3,
            "три": 3, "четверо": 4, "четыре": 4, "пять": 5, "пятеро": 5,
            "шесть": 6, "семь": 7, "восемь": 8, "много": 6, "группа": 5,
            "толпа": 8, "несколько": 3}
INSIDE = ["помещени", "внутри", "магазин", "кафе", "комнат", "зал", "дом",
          "квартир", "коридор", "лифт", "туалет", "раздевал"]
CAR = ["машин", "автомобил", "салон", "за рулём", "водител", "буханк"]
OUT = ["улиц", "улич", "двор", "парк", "дорог", "площад", "поле", "лес",
       "набережн", "стадион", "асфальт"]


def load_desc(path):
    d = json.load(open(path))
    keys = sorted(d, key=lambda k: int(re.findall(r"\d+", k)[0]))
    return [d[k] for k in keys]


def build_vocab(paths, top=120):
    c = Counter()
    for p in paths:
        for t in load_desc(p):
            c.update(set(re.findall(r"[а-яё]{4,}", t.lower())))
    stop = {"стрим", "уличный", "кадр", "видео", "фоне", "камер", "человек",
            "человека", "людей"}
    return [w for w, _ in c.most_common(top * 2) if w not in stop][:top]


def features(blocks, desc, vocab):
    n = len(blocks)
    m = len(desc)
    low = [t.lower() for t in desc]
    wsets = [set(re.findall(r"[а-яё]{4,}", t)) for t in low]

    # к какому кадру относится блок
    spans = []
    for b in blocks:
        i0 = int(b["start"] / SEC)
        i1 = int(math.ceil(b["end"] / SEC))
        i0 = max(0, min(i0, m - 1)); i1 = max(i0 + 1, min(i1, m))
        spans.append((i0, i1))

    V = np.zeros((n, len(vocab)), np.float32)
    people = np.zeros(n, np.float32)
    place = np.zeros((n, 3), np.float32)
    for k, (i0, i1) in enumerate(spans):
        cnt = Counter()
        for i in range(i0, i1):
            cnt.update(wsets[i])
        L = max(i1 - i0, 1)
        for j, w in enumerate(vocab):
            V[k, j] = cnt[w] / L
        ppl = []
        for i in range(i0, i1):
            got = [NUMWORDS[w] for w in wsets[i] if w in NUMWORDS]
            mm = re.findall(r"\b(\d)\s*(?:человек|людей|парн|мужчин)", low[i])
            got += [int(x) for x in mm]
            if got:
                ppl.append(max(got))
        people[k] = np.mean(ppl) if ppl else 0.0
        for i in range(i0, i1):
            t = low[i]
            place[k, 0] += any(x in t for x in INSIDE)
            place[k, 1] += any(x in t for x in CAR)
            place[k, 2] += any(x in t for x in OUT)
        place[k] /= L

    # новизна описания: доля слов, которых раньше в этом стриме не было
    seen = Counter()
    nov_all = np.zeros(m, np.float32)
    for i in range(m):
        w = wsets[i]
        nov_all[i] = (sum(1 for x in w if seen[x] == 0) / len(w)) if w else 0.0
        for x in w:
            seen[x] += 1
    # похожесть на кадр пять минут назад — то же ли это место
    same_prev = np.zeros(m, np.float32)
    for i in range(m):
        j = max(0, i - 25)
        a, b = wsets[i], wsets[j]
        same_prev[i] = len(a & b) / max(len(a | b), 1)
    # длина однородной зоны: пока кадры похожи на предыдущий
    zone = np.zeros(m, np.float32)
    run = 0
    for i in range(1, m):
        sim = len(wsets[i] & wsets[i - 1]) / max(len(wsets[i] | wsets[i - 1]), 1)
        run = run + 1 if sim >= 0.4 else 0
        zone[i] = min(run, 60)

    nov = np.zeros(n, np.float32); sp = np.zeros(n, np.float32)
    zn = np.zeros(n, np.float32)
    for k, (i0, i1) in enumerate(spans):
        nov[k] = nov_all[i0:i1].mean()
        sp[k] = same_prev[i0:i1].mean()
        zn[k] = zone[i0:i1].mean()

    X = np.column_stack([V, people, place, nov, sp, zn]).astype(np.float32)
    names = ([f"desc_{w}" for w in vocab] + ["desc_people", "desc_inside",
             "desc_car", "desc_out", "desc_novelty", "desc_same5min",
             "desc_zone"])
    return X, names
