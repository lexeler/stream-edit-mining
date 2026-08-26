#!/usr/bin/env python3
"""
Лента элементов первых 20 минут для языковой модели — с таймкодами.

Отличие от прежнего подхода (зоны по 30 с с усреднённым текстом): здесь
модель видит то же, что видит монтажёр — последовательность реплик и пауз
с точными таймкодами и длительностями, и решает по каждому элементу.

Окна по 5 минут со сдвигом 2,5 минуты: каждый элемент попадает в два окна,
оценки усредняются. Это убирает краевые эффекты и снижает шум.

ВАЖНО про утечку. Прошлая попытка провалилась, потому что инструкция
описывала устройство начала эфира, а сама была выведена из разбора слотов
1,3,4,5,6 — на них замер завышался, на непросмотренных эфирах эффекта
не было. Поэтому здесь знание передаётся НЕ описанием правил, а примерами
размеченных начал из ДРУГИХ слотов: для целевого слота X примеры берутся
только из слотов, отличных от X.

  ./elems_make.py --slot 2 --examples 4,6
"""
import json, os, argparse

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
PREF = os.environ.get("PREF", "")
OUT = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "elems")
LIMIT = 1200.0
WIN = float(os.environ.get("WIN", 300.0))
STEP = float(os.environ.get("STEP", 150.0))


def mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def line(b, with_answer=False):
    t = mmss(b["start"])
    au = ", ".join(b.get("au", [])[:3])
    if b["kind"] == "speech":
        d = len(b["text"]) / max(b["dur"], 0.3)
        mark = " РВАНО" if d < 6 else (" ПЛОТНО" if d >= 14 else "")
        s = (f"#{b['i']} {t} реч {b['dur']:.1f}с{mark}"
             + (f" [{au}]" if au else "") + f" {b['text']}")
    else:
        s = f"#{b['i']} {t} пауза {b['dur']:.1f}с" + (f" [{au}]" if au else "")
    if with_answer:
        u = b["used_frac"]
        s += f"   ==> {'ВЗЯЛ' if u > 0.5 else 'выбросил'}"
    return s


def blocks(slot, limit=LIMIT):
    B = json.load(open(f"{DATA}/s{slot}.blocks.json"))["blocks"]
    return [b for b in B if b["start"] < limit]


def example_text(slot):
    B = blocks(slot)
    used = sum(b["dur"] * b["used_frac"] for b in B)
    tot = sum(b["dur"] for b in B)
    head = (f"=== ПРИМЕР: другой эфир этого же канала, первые 20 минут.\n"
            f"=== Монтажёр взял отсюда {used/60:.1f} минут из {tot/60:.0f} "
            f"({used/tot:.0%}). Справа помечено его решение.\n")
    return head + "\n".join(line(b, True) for b in B) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--examples", default="", help="слоты-примеры через запятую")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ex = ""
    if a.examples:
        for e in a.examples.split(","):
            assert e != a.slot, "пример не может быть из целевого слота"
            ex += example_text(e) + "\n"
        open(f"{OUT}/examples_for_s{a.slot}.txt", "w").write(ex)

    B = blocks(a.slot)
    n = 0
    t = 0.0
    while t < LIMIT:
        lo, hi = t, min(t + WIN, LIMIT)
        cur = [b for b in B if lo <= b["start"] < hi]
        if cur:
            ctx_before = [b for b in B if b["start"] < lo][-6:]
            ctx_after = [b for b in B if b["start"] >= hi][:6]
            body = []
            if ctx_before:
                body.append("--- что было перед этим куском (для понимания, "
                            "оценки не давать):")
                body += [line(b) for b in ctx_before]
            body.append(f"--- ОЦЕНИТЬ ЭТИ ЭЛЕМЕНТЫ ({mmss(lo)}–{mmss(hi)}), "
                        f"их {len(cur)} штук:")
            body += [line(b) for b in cur]
            if ctx_after:
                body.append("--- что будет дальше (для понимания, оценки "
                            "не давать):")
                body += [line(b) for b in ctx_after]
            p = f"{OUT}/{PREF}s{a.slot}_w{n:02d}.txt"
            open(p, "w").write("\n".join(body) + "\n")
            ids = [b["i"] for b in cur]
            json.dump(ids, open(f"{OUT}/{PREF}s{a.slot}_w{n:02d}.ids.json", "w"))
            n += 1
        t += STEP
    print(f"слот {a.slot}: {len(B)} элементов, {n} окон по {WIN/60:.0f} мин "
          f"со сдвигом {STEP/60:.1f}"
          + (f", примеры из слотов {a.examples}" if a.examples else ""))


if __name__ == "__main__":
    main()
