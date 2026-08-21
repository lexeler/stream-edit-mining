#!/usr/bin/env python3
"""
Сборка монтажного листа из оценок макро-модели.

Модель отвечает «насколько это стоит оставить». Сборка решает, как из оценок
получить смотрибельный выпуск. Два уровня, как у монтажёра:

  макро — какие области эфира выживают. Витерби по окнам, штраф за
          переключение, доля выбирается самой моделью;
  микро — внутри выживших областей выбрасываем паузы, как это делает он.
          Порог берётся из замера micro.py, а не с потолка.

Плюс правила выпуска: начало в первые 1,5 минуты (7 слотов из 7), резы
только в тишине (99 % склеек монтажёра приходятся на тишину), минимальная
длина куска, склейка близких.
"""
import json, collections, os
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MIN_SEG = 1.2        # куски короче — выбросить
GAP_MERGE = 0.5      # разрыв меньше — склеить
START_LIMIT = 90.0   # первый кусок обязан начаться не позже
END_TAIL = 25.0      # последние секунды эфира оставляем всегда: прощание
# Паузы внутри сцен почти не режем. Замер micro.py: монтажёр выбрасывает
# паузу 1,2–2 с лишь в 28 % случаев, 3–5 с — в 33 %. Правило «резать паузы
# длиннее секунды» было бы неверным в двух случаях из трёх.
PAUSE_DROP = 8.0     # выбрасываем только откровенно мёртвый воздух


def viterbi(sc, lam):
    n = len(sc)
    d0 = np.zeros(n); d1 = np.zeros(n); bk = np.zeros((n, 2), np.int8)
    d0[0], d1[0] = 0.0, sc[0]
    for i in range(1, n):
        a, b = d0[i - 1], d1[i - 1] - lam
        d0[i], bk[i, 0] = (b, 1) if b > a else (a, 0)
        a, b = d1[i - 1], d0[i - 1] - lam
        d1[i], bk[i, 1] = (b + sc[i], 0) if b > a else (a + sc[i], 1)
    st = 1 if d1[-1] > d0[-1] else 0
    out = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        out[i] = bool(st); st = bk[i, st]
    return out


def macro_keep(p, wsec, rate, lam):
    """булева маска окон: доля оставленного = rate, штраф за переключение lam"""
    r = np.empty(len(p)); r[np.argsort(p)] = np.arange(len(p))
    r = r / max(len(p) - 1, 1)
    lo, hi = -1.0, 2.0
    keep = None
    for _ in range(24):
        mid = (lo + hi) / 2
        keep = viterbi(r - mid, lam)
        got = keep.mean()
        if got > rate:
            lo = mid
        else:
            hi = mid
        if abs(got - rate) < 0.002:
            break
    return keep


def build(rows, p, wsec, rate, lam, rms=None, sr100=100, pause_drop=PAUSE_DROP):
    """макро-маска окон -> монтажный лист в секундах"""
    dur = np.array([r["dur"] for r in rows], np.float32)
    st = np.array([r["start"] for r in rows], np.float32)
    isw = np.array([r["is_word"] for r in rows], np.int8)
    T = float(st[-1] + dur[-1])
    keep_w = macro_keep(p, wsec, rate, lam)
    wid = np.clip(((st + dur / 2) / wsec).astype(int), 0, len(keep_w) - 1)
    keep = keep_w[wid].copy()

    # правило начала: первый оставленный кусок не позже START_LIMIT
    first = int(np.argmax(keep)) if keep.any() else 0
    if st[first] > START_LIMIT:
        j = int(np.searchsorted(st, START_LIMIT))
        best = j
        w0 = wid[:j + 1]
        if len(w0):
            best = int(np.argmax(p[w0]))
        keep[best:best + 1] = True
        k = best
        while k + 1 < len(keep) and st[k + 1] < st[best] + 20:
            keep[k + 1] = True; k += 1

    # правило концовки: монтаж всегда доходит до конца эфира (6 слотов из 6,
    # расхождение 0,0–0,3 мин). Прощание сохраняем всегда
    end_from = T - END_TAIL
    tail = st + dur / 2 >= end_from
    if tail.any() and not keep[tail].any():
        j = int(np.argmax(tail))
        keep[j:] = True

    # микро: внутри оставленного выбрасываем длинные паузы
    for i in range(len(rows)):
        if keep[i] and not isw[i] and dur[i] > pause_drop:
            keep[i] = False

    # собираем отрезки
    segs = []
    for i in range(len(rows)):
        if not keep[i]:
            continue
        a, b = float(st[i]), float(st[i] + dur[i])
        if segs and a - segs[-1][1] < 1e-6:
            segs[-1][1] = b
        else:
            segs.append([a, b])

    # склейка близких и выброс обрывков
    m = []
    for a, b in segs:
        if m and a - m[-1][1] <= GAP_MERGE:
            m[-1][1] = b
        else:
            m.append([a, b])
    m = [s for s in m if s[1] - s[0] >= MIN_SEG]

    # притягивание резов к тишине
    if rms is not None:
        R = int(0.5 * sr100)
        out = []
        for a, b in m:
            for which in (0, 1):
                t = a if which == 0 else b
                lo = max(0, int(t * sr100) - R); hi = min(len(rms), int(t * sr100) + R)
                if hi > lo:
                    q = (lo + int(np.argmin(rms[lo:hi]))) / sr100
                    if which == 0:
                        a = q
                    else:
                        b = q
            if b - a >= MIN_SEG:
                out.append([a, b])
        m = []
        for a, b in out:
            if m and a < m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in m], T


def build_to_duration(rows, p, wsec, target_sec, lam, rms=None, sr100=100,
                      pause_drop=PAUSE_DROP):
    """Собрать лист заданной ДЛИТЕЛЬНОСТИ, а не заданной доли окон.

    Если подтягивать паузы внутри сцен, выпуск укорачивается — и мы теряем
    покрытие просто так. Правильно взять взамен больше сцен, чтобы
    хронометраж остался тем же. Подбираем долю окон под нужную длительность.
    """
    lo, hi = 0.02, 0.98
    best = None
    for _ in range(18):
        mid = (lo + hi) / 2
        segs, T = build(rows, p, wsec, mid, lam, rms, sr100, pause_drop)
        got = sum(b - a for a, b in segs)
        best = (segs, T)
        if got < target_sec:
            lo = mid
        else:
            hi = mid
        if abs(got - target_sec) < 0.005 * target_sec:
            break
    return best


def evaluate(segs, rows):
    """совпадение по времени с настоящим монтажом"""
    dur = np.array([r["dur"] for r in rows], np.float32)
    st = np.array([r["start"] for r in rows], np.float32)
    frac = np.array([r["used_frac"] for r in rows], np.float32)
    mid = st + dur / 2
    sel = np.zeros(len(rows), bool)
    for a, b in segs:
        sel |= (mid >= a) & (mid < b)
    ov = float((sel * frac * dur).sum())
    pred = float(sum(b - a for a, b in segs))
    true = float((frac * dur).sum())
    p = ov / pred if pred else 0
    r = ov / true if true else 0
    return dict(точность=round(p, 3), полнота=round(r, 3),
                F1=round(2 * p * r / (p + r), 3) if p + r else 0,
                IoU=round(ov / (pred + true - ov), 3),
                минут=round(pred / 60, 1), кусков=len(segs))
