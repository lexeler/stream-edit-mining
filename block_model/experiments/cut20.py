#!/usr/bin/env python3
"""
Нарезка первых 20 минут эфира с правильным открытием выпуска.

Требование заказчика: выпуск обязан начинаться с приветствия. Само
приветствие ищет `startpoint.py` (языковая модель отличает «здорово,
парни» от «это здорово»), а здесь оно превращается в точку реза:

  1. приветствие часто сидит в СЕРЕДИНЕ длинной реплики — тогда режем
     внутри неё: позиция фразы оценивается по доле текста и притягивается
     к ближайшему промежутку между словами;
  2. всё, что было до приветствия, выбрасывается целиком, каким бы
     «ценным» его ни считала модель: выпуск не может открываться с середины
     разговора;
  3. завязка (приветствие → кто в кадре → план дня) берётся непрерывно,
     чтобы её не порвало на полуслове;
  4. дальше идёт обычный отбор по оценке модели.

  ./cut20.py --slot 2
"""
import json, os, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble as A

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
AUD = os.path.join(os.environ.get("SEM_DATASET", "work/dataset"), "звук")
BASE = os.environ.get("SEM_STREAMS", "work/streams")
LIMIT = 1200.0


def rank01(v):
    r = np.empty(len(v), float)
    r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def greet_time(slot, blocks):
    """время, с которого звучит приветствие"""
    p = f"{DATA}/startpoint/s{slot}.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    st = int(d["старт"]); en = int(d.get("конец_завязки", st))
    bs = {b["i"]: b for b in blocks}
    b = bs.get(st)
    if b is None:
        return None, None
    t0 = float(b["start"])
    q = str(d.get("цитата", "")).strip()
    words = b["text"].split()
    if q and q.lower() != "нет" and len(words) > 3:
        # номер слова, с которого начинается приветствие
        head = q.split()[0].lower().strip(".,!?»«\"'")
        k = next((i for i, w in enumerate(words)
                  if w.lower().strip(".,!?»«\"'").startswith(head[:6])), None)
        ip = f"{DATA}/items/s{slot}.items.json"
        if k is not None and os.path.exists(ip):
            it = json.load(open(ip))
            snd = [r for r in it if r["is_word"]
                   and r["start"] >= b["start"] - 0.01
                   and r["end"] <= b["end"] + 0.01]
            if snd:
                # в ленте одно звуковое событие на слово; если счёт разошёлся,
                # отображаем пропорционально
                j = k if len(snd) == len(words) else int(
                    round(k * len(snd) / max(len(words), 1)))
                j = max(0, min(j, len(snd) - 1))
                t0 = float(snd[j]["start"]) - 0.15   # чтобы не срезать первый слог
    return max(t0, float(b["start"])), en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2")
    ap.add_argument("--scores", default="")
    ap.add_argument("--density", type=float, default=7.06)
    a = ap.parse_args()

    B0 = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    B = [b for b in B0 if b["start"] < LIMIT]
    f = a.scores or f"{DATA}/final3_s{a.slot}.npy"
    d = np.load(f, allow_pickle=True).item()
    m = d["start"] < LIMIT
    y, dur, start = d["y"][m], d["dur"][m], d["start"][m]
    p = rank01(rank01(d["pb"])[m])
    kind = np.array([1 if b["kind"] == "speech" else 0 for b in B])

    t0, en = greet_time(a.slot, B)
    if t0 is None:
        print("точка приветствия не найдена — сначала запусти startpoint.py")
        return
    bs = {b["i"]: b for b in B}
    t_end = float(bs[en]["end"]) if en in bs else t0 + 25.0
    t_end = max(t_end, t0 + 12.0)
    print(f"слот {a.slot}: выпуск открывается в "
          f"{int(t0)//60}:{int(t0)%60:02d}, завязка до "
          f"{int(t_end)//60}:{int(t_end)%60:02d}")

    # всё до приветствия под запретом
    p = np.where(start + dur <= t0, -1.0, p)
    target = float((dur * y).sum())

    lo, hi = 0.0, 0.6
    for _ in range(18):
        kp = (lo + hi) / 2
        segs = A.build_to_duration(B, p, dur, start, kind, target, kp, 80.0)
        got = len(segs) / max(sum(e - s for s, e in segs) / 60, 1e-6)
        if got < a.density:
            lo = kp
        else:
            hi = kp
    segs = A.build_to_duration(B, p, dur, start, kind, target, (lo + hi) / 2, 80.0)

    # завязка идёт целиком, и выпуск начинается ровно с приветствия
    segs = [s for s in segs if s[1] > t_end]
    segs = [[t0, t_end]] + segs
    fl = f"{AUD}/s{a.slot}.flac"
    if os.path.exists(fl):
        rms = A.rms_envelope(fl)
        tail = A.snap_to_silence(segs[1:], rms)
        segs = [segs[0]] + tail
    segs = [s for s in segs if s[1] <= LIMIT and s[1] - s[0] >= 1.0]
    segs = sorted(segs)

    mm = A.evaluate(segs, B, dur, y, start)
    print(f"наша нарезка: {mm['кусков']} кусков, {mm['минут']:.1f} мин, "
          f"медиана {mm['медиана']:.1f} с")
    print(f"ТОЧНОСТЬ {mm['точность']:.3f}  полнота {mm['полнота']:.3f}")

    tc = json.load(open(f"{BASE}/{a.slot}/narezki/timecodes.json"))["segments"]
    ed = [[e["start"], min(e["end"], LIMIT)] for e in tc if e["start"] < LIMIT]
    print(f"у монтажёра там: {len(ed)} кусков, "
          f"{sum(b-a_ for a_,b in ed)/60:.1f} мин, начало "
          f"{int(ed[0][0])//60}:{int(ed[0][0])%60:02d}")

    out = f"{DATA}/edl_s{a.slot}_first20.json"
    json.dump({"наша": [[round(s, 3), round(e, 3)] for s, e in segs],
               "монтажёр": [[round(s, 3), round(e, 3)] for s, e in ed],
               "метрики": mm}, open(out, "w"), ensure_ascii=False)
    print(f"монтажные листы: {out}")


if __name__ == "__main__":
    main()
