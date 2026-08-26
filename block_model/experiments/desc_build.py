#!/usr/bin/env python3
"""
Признаки из описаний кадров на сетке 12 с — для прежнего оконного конвейера.

Он сильнее блочного (0,559 против 0,523 на слоте 7), поэтому новый канал
подключаем прежде всего к нему. Описания даёт видео-языковая модель, одна
строка на кадр раз в 12 секунд.

Словарь частых слов строится ТОЛЬКО по обучающим слотам, тестовый в него
не заглядывает.

  python3 desc_build.py
Кладёт $SEM_WORK/data/desc_s<N>.npz (feats, names, hop).
"""
import json, os, re, math
from collections import Counter
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SEC = 12.0
TRAIN = ["1", "3", "4", "5", "6"]
ALL = ["1", "2", "3", "4", "5", "6", "7"]
TOP = 140
NUMWORDS = {"один": 1, "одна": 1, "двое": 2, "два": 2, "две": 2, "трое": 3,
            "три": 3, "четверо": 4, "четыре": 4, "пять": 5, "пятеро": 5,
            "шесть": 6, "семь": 7, "восемь": 8, "много": 6, "группа": 5,
            "толпа": 8, "несколько": 3}
INSIDE = ["помещени", "внутри", "магазин", "кафе", "комнат", "зал", "дом",
          "квартир", "коридор", "лифт", "раздевал", "столом", "кухн"]
CAR = ["машин", "автомобил", "салон", "рулём", "водител", "буханк", "едут"]
OUT = ["улиц", "улич", "двор", "парк", "дорог", "площад", "поле", "лес",
       "набережн", "стадион", "асфальт", "трав"]
STOP = {"стрим", "уличный", "кадр", "видео", "фоне", "камер", "человек",
        "человека", "людей", "человеком"}


def load(slot):
    p = f"{DATA}/desc_s{slot}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    keys = sorted(d, key=lambda k: int(re.findall(r"\d+", k)[0]))
    return [d[k] for k in keys]


def main():
    vocab_c = Counter()
    for s in TRAIN:
        t = load(s)
        if not t:
            continue
        for x in t:
            vocab_c.update(set(re.findall(r"[а-яё]{4,}", x.lower())))
    vocab = [w for w, _ in vocab_c.most_common(TOP * 2) if w not in STOP][:TOP]
    print(f"словарь из {len(vocab)} слов, построен по слотам {'+'.join(TRAIN)}")

    for s in ALL:
        desc = load(s)
        if not desc:
            print(f"слот {s}: описаний нет, пропуск")
            continue
        m = len(desc)
        low = [x.lower() for x in desc]
        wsets = [set(re.findall(r"[а-яё]{4,}", x)) for x in low]

        V = np.zeros((m, len(vocab)), np.float32)
        for i, ws in enumerate(wsets):
            for j, w in enumerate(vocab):
                if w in ws:
                    V[i, j] = 1.0
        people = np.zeros(m, np.float32)
        place = np.zeros((m, 3), np.float32)
        for i in range(m):
            got = [NUMWORDS[w] for w in wsets[i] if w in NUMWORDS]
            got += [int(x) for x in re.findall(
                r"\b(\d)\s*(?:человек|людей|парн|мужчин)", low[i])]
            people[i] = max(got) if got else 0
            place[i, 0] = any(x in low[i] for x in INSIDE)
            place[i, 1] = any(x in low[i] for x in CAR)
            place[i, 2] = any(x in low[i] for x in OUT)

        seen = Counter(); nov = np.zeros(m, np.float32)
        for i in range(m):
            w = wsets[i]
            nov[i] = (sum(1 for x in w if seen[x] == 0) / len(w)) if w else 0.0
            for x in w:
                seen[x] += 1
        # то же ли это место, что минуту / пять минут назад
        same1 = np.zeros(m, np.float32); same5 = np.zeros(m, np.float32)
        for i in range(m):
            for lag, arr in ((5, same1), (25, same5)):
                j = max(0, i - lag)
                a, b = wsets[i], wsets[j]
                arr[i] = len(a & b) / max(len(a | b), 1)
        # длина однородной зоны и сколько раз такое место уже было
        zone = np.zeros(m, np.float32); run = 0
        for i in range(1, m):
            sim = len(wsets[i] & wsets[i - 1]) / max(len(wsets[i] | wsets[i - 1]), 1)
            run = run + 1 if sim >= 0.4 else 0
            zone[i] = min(run, 60)
        seen_places = np.zeros(m, np.float32)
        for i in range(m):
            if i < 10:
                continue
            k = 0
            for j in range(0, i - 10):
                a, b = wsets[i], wsets[j]
                if len(a & b) / max(len(a | b), 1) >= 0.5:
                    k += 1
            seen_places[i] = min(k, 200)

        F = np.column_stack([V, people, place, nov, same1, same5, zone,
                             seen_places]).astype(np.float32)
        names = ([f"кадр:{w}" for w in vocab] +
                 ["кадр:людей", "кадр:внутри", "кадр:машина", "кадр:улица",
                  "кадр:новизна", "кадр:то же 1мин", "кадр:то же 5мин",
                  "кадр:длина зоны", "кадр:было раньше"])
        np.savez(f"{DATA}/desc_s{s}.npz", feats=F, names=np.array(names, object),
                 hop=SEC)
        print(f"слот {s}: {m} кадров, {F.shape[1]} признаков")


if __name__ == "__main__":
    main()
