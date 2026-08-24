#!/usr/bin/env python3
"""
Разметка реплик по смыслу: языковая модель как размечающий, а не как судья.

Судьёй она проигрывает бустингу (0,44 против 0,56), и в смеси не добавляет
ничего: она судит по теме, а тема внутри сцены не предсказывает решение
(51,8 % на парном контроле — ровно монетка). Зато она умеет то, чего нет
ни в звуке, ни в акустике: сказать, ЧТО СЕЙЧАС ПРОИСХОДИТ — едут в машине,
играют, обслуживают чат, фотографируются с фанатом, ждут.

Веса этих типов считаются потом по размеченным слотам, а не задаются
на глаз. Модель отвечает буквами — выход короткий, ошибиться трудно.

  mark.py --slot 7 --tag m1 --chunk 150
"""
import json, os, re, sys, time, argparse
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import call, render_lines, DATA, MODEL

TYPES = {
    "A": "активность: герои прямо сейчас чем-то заняты в кадре — играют, "
         "лазают, стреляют, катаются, участвуют в чужом деле, что-то "
         "пробуют, показывают",
    "G": "разговор с посторонним человеком (продавец, прохожий, случайный "
         "знакомый), где есть содержание, а не только вежливость",
    "R": "результат или реакция сразу после события: счёт, «забил», "
         "«получилось», возглас, оценка только что случившегося",
    "P": "подготовка и организация внутри занятия: «давай ещё раз», кто где "
         "встанет, правила во второй раз, снаряжение, настройка",
    "D": "переезд: едут в машине, идут между местами, дорога",
    "W": "ожидание и бытовая возня: ждут, ищут, собираются, платят, "
         "переодеваются, выбирают товар",
    "T": "трёп между своими не по делу: деньги, заработки, тачки, планы "
         "на завтра, воспоминания, обсуждение отсутствующих",
    "S": "обслуживание стрима: чат, ники, донаты, промокоды, реклама, "
         "ставки, техника, связь, «зайдите через браузер»",
    "F": "фан-сервис: фото с прохожими, приветы по именам, автографы",
    "X": "невнятное, шум, обрывки без смысла, молчание",
}

SYSTEM = """Ты размечаешь расшифровку уличного стрима блогера — он ходит и ездит по городу с друзьями.

Каждой реплике поставь ОДНУ букву — что происходит в этот момент:

""" + "\n".join(f"  {k} — {v}" for k, v in TYPES.items()) + """

И при необходимости добавь пометки сразу после буквы:
  !  — смешно, сильная эмоция, ругань на эмоциях, крик восторга
  r  — это повтор: такое же занятие, тема или шутка уже были в этом стриме
  n  — это происходит впервые: новое место, новый человек, новая тема

Смотри на происходящее, а не на отдельные слова. Реплики вокруг — часть
той же сцены, размечай их одинаково, пока сцена не сменилась.

ФОРМАТ ОТВЕТА — строго строки «номер:буква» или «номер-номер:буква»,
соседние одинаковые объединяй в диапазоны. Никаких пояснений.

Пример:
12-40:D
41-52:An
53:R!
54-60:Pr"""

USER = """Расшифровка стрима, реплики {a}–{b} (время {t0}–{t1}).

{text}

Размести реплики {a}–{b} по типам."""


def parse(txt, lo, hi):
    out = {}
    for m in re.finditer(r"(\d+)\s*(?:[-–—]\s*(\d+))?\s*:\s*([AGRPDWTSFX])([!rn]*)",
                         txt):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        t, fl = m.group(3), m.group(4) or ""
        if b < a or b - a > 3000:
            continue
        for i in range(max(a, lo), min(b, hi) + 1):
            out[i] = t + "".join(sorted(set(fl)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    blocks = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    lines, mapping = render_lines(blocks, with_audio=True)
    N = len(mapping)
    bounds = [(i + 1, min(i + a.chunk, N)) for i in range(0, N, a.chunk)]

    def mmss(t):
        return f"{int(t)//60}:{int(t)%60:02d}"

    def one(k):
        lo, hi = bounds[k]
        ctx = max(1, lo - 15)
        sel = []
        for num, s in lines:
            if num is None:
                if sel:
                    sel.append(s)
                continue
            if ctx <= num <= hi:
                sel.append(s)
        b0 = blocks[mapping[lo]]; b1 = blocks[mapping[hi]]
        user = USER.format(a=lo, b=hi, text="\n".join(sel),
                           t0=mmss(b0["start"]), t1=mmss(b1["end"]))
        try:
            txt, u = call(a.model, SYSTEM, user, temperature=0.0,
                          max_tokens=6000, reasoning=False)
        except Exception as e:
            print(f"  кусок {k+1}: ПРОВАЛ {e}", flush=True)
            return {"k": k, "marks": {}, "usage": {}}
        mk = parse(txt, lo, hi)
        print(f"  кусок {k+1}/{len(bounds)} ({lo}-{hi}): {len(mk)}/{hi-lo+1}",
              flush=True)
        return {"k": k, "marks": mk, "usage": u}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, range(len(bounds))))
    local = {}
    for r in res:
        local.update(r["marks"])
    marks = {mapping[k]: v for k, v in local.items() if k in mapping}
    pi = sum(r["usage"].get("prompt_tokens", 0) for r in res)
    po = sum(r["usage"].get("completion_tokens", 0) for r in res)
    cost = pi / 1e6 * 0.07 + po / 1e6 * 0.13
    os.makedirs(f"{DATA}/marks", exist_ok=True)
    out = f"{DATA}/marks/s{a.slot}_{a.tag}.json"
    json.dump({"slot": a.slot, "model": a.model,
               "marks": {str(k): v for k, v in marks.items()},
               "cost_usd": round(cost, 4)}, open(out, "w"), ensure_ascii=False)
    from collections import Counter
    c = Counter(v[0] for v in marks.values())
    print(f"\n{len(bounds)} запросов за {time.time()-t0:.0f} с, ${cost:.4f}, "
          f"покрыто {len(local)}/{N}")
    print("типы: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
