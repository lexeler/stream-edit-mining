#!/usr/bin/env python3
"""
Сборка готовой нарезки — в манере самого монтажёра.

Замер по семи выпускам: у него 405–601 кусок на стрим, медиана куска
4–9 секунд, максимум 56–140 с, а медиана ВЫБРОШЕННОГО промежутка — 1,0–1,8 с.
То есть он не берёт сцены монолитом, а постоянно подчищает: убирает паузы,
обрывки, провисания внутри живых сцен.

Прежняя сборка со штрафом за переключение давала 42 куска по минуте и
больше — это не его работа, это черновая разметка зон. Здесь порядок другой:

  1. отбираем блоки по оценке модели, без сглаживания в монолиты;
  2. внутри оставленного выбрасываем паузы длиннее порога — от этого выпуск
     становится плотным, как у него;
  3. хронометраж от подчистки падает, поэтому добираем ещё материала,
     пока не вернёмся к нужной длине (иначе подтяжка достаётся ценой
     покрытия — это уже проверялось прежними опытами);
  4. резы притягиваем к самой тихой точке рядом: склеек «речь|речь»
     у монтажёра всего 1,5 %.

  assemble.py --slot 2 --pause 1.0
"""
import json, os, argparse, subprocess
import numpy as np

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
AUD = os.path.join(os.environ.get("SEM_DATASET", "work/dataset"), "звук")
MIN_SEG = 1.0        # куски короче — выбросить
SNAP = 0.4           # радиус притягивания реза к тишине, с
START_LIMIT = 90.0
END_TAIL = 20.0


def rms_envelope(path, sr=8000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path,
                          "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    W = sr // 100
    n = len(x) // W
    return np.sqrt((x[:n * W].reshape(n, W) ** 2).mean(1))


def build(B, p, dur, start, kind, thr, pause_thr, min_seg=MIN_SEG):
    """отбор блоков по оценке; паузы отбираются по СВОЕЙ, более строгой планке.

    Монтажёр внутри живой сцены вырезает каждую третью паузу — от этого
    выпуск и получается из 400–600 кусков. По какой именно, модель угадывает
    с AUC 0,59–0,77, а правило «короткая значит оставить» — только 0,43–0,67.
    Поэтому паузы режем по оценке модели, а не по длине.
    """
    keep = np.where(kind == 1, p >= thr, p >= pause_thr)
    # Правило начала: выпуск стартует в первые полторы минуты эфира.
    # Начинать надо с РЕЧИ: в начале стрима обычно длинная тишина, пока
    # хозяин проверяет, пошёл ли эфир, и если взять её, кусок уйдёт впустую
    # (на слоте 7 так терялось 112 секунд).
    early = np.flatnonzero((start < START_LIMIT) & (kind == 1))
    if not len(early):
        # бывает, что первые полторы минуты эфира — сплошная тишина: хозяин
        # проверяет, пошла ли трансляция. Тогда цепляться не за что, и без
        # этой ветки выпуск начинался с девятой минуты (слот 1). Берём
        # первую речь эфира, какой бы поздней она ни была.
        sp = np.flatnonzero(kind == 1)
        early = sp[:1]
    if len(early) and not keep[start < START_LIMIT].any():
        j = int(early[np.argmax(p[early])])
        k = j
        while k < len(keep) and start[k] < start[j] + 20:
            keep[k] = True
            k += 1
    T = float(start[-1] + dur[-1])
    tail = start + dur >= T - END_TAIL
    if tail.any() and not keep[tail].any():
        keep[np.argmax(tail):] = True
    segs = []
    for i, b in enumerate(B):
        if not keep[i]:
            continue
        s_, e_ = float(b["start"]), float(b["end"])
        if segs and s_ - segs[-1][1] < 1e-6:
            segs[-1][1] = e_
        else:
            segs.append([s_, e_])
    return [x for x in segs if x[1] - x[0] >= min_seg]


def split_long(segs, B, p, dur, start, kind, maxseg):
    """дробим слишком длинные куски: у монтажёра самый длинный кусок 56–140 с.

    Режем по паузе с наименьшей оценкой внутри куска — там, где модель менее
    всего уверена. Если паузы внутри нет, кусок остаётся: резать сплошную
    речь наугад хуже, чем оставить её целой.
    """
    out = []
    queue = list(segs)
    while queue:
        s_, e_ = queue.pop(0)
        if e_ - s_ <= maxseg:
            out.append([s_, e_])
            continue
        inside = [i for i in range(len(B))
                  if kind[i] == 0 and start[i] > s_ + 8 and start[i] + dur[i] < e_ - 8]
        if not inside:
            out.append([s_, e_])
            continue
        j = inside[int(np.argmin(p[inside]))]
        a1, b1 = s_, float(start[j])
        a2, b2 = float(start[j] + dur[j]), e_
        for seg in ((a1, b1), (a2, b2)):
            if seg[1] - seg[0] >= MIN_SEG:
                queue.append([seg[0], seg[1]])
    return sorted(out)


def build_to_duration(B, p, dur, start, kind, target, pause_keep, maxseg=0):
    """подбираем планку для речи так, чтобы вернуть нужный хронометраж.

    pause_keep здесь — НАДБАВКА к планке для пауз (в единицах ранга оценки).
    Ноль означает «паузы наравне с речью» и даёт монолитные куски; чем больше
    надбавка, тем чаще тишина внутри сцены вырезается и тем мельче нарезка.
    """
    lo, hi = float(p.min()) - 1e-6, float(p.max()) + 1e-6
    best = None
    for _ in range(32):
        thr = (lo + hi) / 2
        # к паузе планка строже на pause_keep: так подчищается тишина
        # внутри взятых сцен, а хронометраж возвращается добором речи
        segs = build(B, p, dur, start, kind, thr, thr + pause_keep)
        if maxseg:
            segs = split_long(segs, B, p, dur, start, kind, maxseg)
        got = sum(e - s_ for s_, e in segs)
        best = segs
        if got > target:
            lo = thr
        else:
            hi = thr
        if abs(got - target) < 0.004 * target:
            break
    return best


def snap_to_silence(segs, rms, r=SNAP):
    out = []
    R = int(r * 100)
    for s, e in segs:
        for which in (0, 1):
            t = s if which == 0 else e
            lo = max(0, int(t * 100) - R)
            hi = min(len(rms), int(t * 100) + R)
            if hi > lo:
                q = (lo + int(np.argmin(rms[lo:hi]))) / 100.0
                if which == 0:
                    s = q
                else:
                    e = q
        if e - s >= MIN_SEG:
            out.append([s, e])
    m = []
    for s, e in out:
        if m and s < m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


def evaluate(segs, B, dur, y, start):
    mid = start + dur / 2
    sel = np.zeros(len(B), bool)
    for s, e in segs:
        sel |= (mid >= s) & (mid < e)
    ov = float((dur * y)[sel].sum())
    pred = float(sum(e - s for s, e in segs))
    true = float((dur * y).sum())
    L = np.array([e - s for s, e in segs]) if segs else np.array([0.0])
    return {"точность": ov / pred if pred else 0, "полнота": ov / true,
            "минут": pred / 60, "кусков": len(segs),
            "медиана": float(np.median(L)), "макс": float(L.max())}


def zone_gate(p, dur, start, kind, judge, W, R1, rate):
    """Ступень 1: отсечь зоны-пустоты, чтобы добор хронометража не размазывался.

    Проверено на шести слотах с вложенным выбором настроек: точность
    не меняется (+0,0001 ± 0,0066), но форма становится ближе к монтажёру —
    340 кусков вместо 473, медиана 6,7 с вместо 4,8 (у него около 6–7).
    Причина: без этого сборка, добирая хронометраж после чистки пауз,
    рассыпает мелкие куски по всему эфиру, включая мёртвые зоны.
    """
    mid = start + dur / 2
    T = float(dur.sum())
    z = (mid // W).astype(int)
    K = int(z.max()) + 1
    zd = np.bincount(z, weights=dur, minlength=K)
    zs = (np.bincount(z, weights=dur * judge, minlength=K)
          / np.maximum(zd, 1e-9))
    o = np.argsort(-zs)
    c = np.cumsum(zd[o])
    zk = np.zeros(K, bool)
    zk[o[c <= R1 * rate * T]] = True
    return np.where(zk[z], p, p.min() - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2")
    ap.add_argument("--src", default="pb", choices=["pb", "mix", "pw"])
    ap.add_argument("--zones", default="", help="ступень 1: W:запас:судья, "
                    "например 30:1.3:pw (пусто — без зонного отсева)")
    ap.add_argument("--keep-pauses", type=float, default=0.1,
                    help="надбавка к планке для пауз: больше — мельче нарезка")
    ap.add_argument("--maxseg", type=float, default=100.0,
                    help="предельная длина куска, с; у монтажёра 56–140")
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--scores", default="", help="файл с оценками; по умолчанию "
                    "data/final_s<слот>.npy")
    ap.add_argument("--density", type=float, default=0.0,
                    help="подобрать надбавку так, чтобы вышло столько кусков "
                         "на минуту выпуска. У монтажёра 7,06 в среднем по "
                         "шести обучающим слотам (разброс 4,1–9,2)")
    ap.add_argument("--sweep", action="store_true", help="перебрать пороги паузы")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    B = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    d = np.load(a.scores or f"{DATA}/final_s{a.slot}.npy",
                allow_pickle=True).item()
    p, y, dur, start = d[a.src], d["y"], d["dur"], d["start"]
    kind = np.array([1 if b["kind"] == "speech" else 0 for b in B])
    T = float(dur.sum())
    rate = a.rate or float(d["rate"])
    target = rate * T

    if a.zones:
        W, R1, jn = a.zones.split(":")
        p = zone_gate(p, dur, start, kind, d[jn], float(W), float(R1), rate)
        print(f"ступень 1: зоны по {W} с, запас x{R1}, судья {jn}")

    fl = f"{AUD}/s{a.slot}.flac"
    rms = rms_envelope(fl) if os.path.exists(fl) else None

    if a.sweep:
        print(f"{'штраф':>7}{'кусков':>8}{'минут':>8}{'медиана':>9}{'макс':>7}"
              f"{'точность':>10}{'полнота':>9}")
        for pd in (0.0, 0.03, 0.06, 0.1, 0.15, 0.22, 0.3, 0.45):
            segs = build_to_duration(B, p, dur, start, kind, target, pd, a.maxseg)
            if rms is not None:
                segs = snap_to_silence(segs, rms)
            m = evaluate(segs, B, dur, y, start)
            print(f"{pd:>7.2f}{m['кусков']:>8}{m['минут']:>8.1f}"
                  f"{m['медиана']:>9.1f}{m['макс']:>7.0f}{m['точность']:>10.3f}"
                  f"{m['полнота']:>9.3f}")
        return

    keep_pauses = a.keep_pauses
    if a.density:
        # надбавка к планке для пауз задаёт, насколько дробной выйдет нарезка.
        # Подбираем её под манеру монтажёра, а не под метрику: чем больше
        # надбавка, тем чаще вырезается тишина внутри взятых сцен.
        lo, hi = 0.0, 0.6
        for _ in range(18):
            mid_kp = (lo + hi) / 2
            s2 = build_to_duration(B, p, dur, start, kind, target, mid_kp,
                                   a.maxseg)
            got = len(s2) / max(sum(e - s_ for s_, e in s2) / 60, 1e-6)
            if got < a.density:
                lo = mid_kp
            else:
                hi = mid_kp
        keep_pauses = (lo + hi) / 2
        print(f"плотность {a.density:.2f} куска на минуту -> надбавка "
              f"{keep_pauses:.3f}")
    segs = build_to_duration(B, p, dur, start, kind, target, keep_pauses,
                             a.maxseg)
    if rms is not None:
        segs = snap_to_silence(segs, rms)
    m = evaluate(segs, B, dur, y, start)
    print(f"слот {a.slot}: выпуск {m['минут']:.1f} мин, "
          f"у монтажёра {(dur*y).sum()/60:.1f} мин")
    print(f"кусков {m['кусков']} (у монтажёра "
          f"{len(json.load(open(f'$SEM_STREAMS/{a.slot}/narezki/timecodes.json'))['segments'])}), "
          f"медиана {m['медиана']:.1f} с, максимум {m['макс']:.0f} с")
    print(f"ТОЧНОСТЬ {m['точность']:.3f}  полнота {m['полнота']:.3f}  "
          f"случайно {(dur*y).sum()/T:.3f}")
    out = a.out or f"{DATA}/edl_s{a.slot}_fine.json"
    json.dump({"slot": a.slot, "rate": rate,
               "segments": [[round(s, 3), round(e, 3)] for s, e in segs],
               "metrics": m}, open(out, "w"), ensure_ascii=False)
    print(f"монтажный лист: {out}")


if __name__ == "__main__":
    main()
