#!/usr/bin/env python3
"""
Края эфира отдельно: начало и конец монтирует языковая модель по своим
инструкциям, середину — обучаемая.

Зачем именно так. Замер по шести эфирам: в первых 10 % точность модели 0,713,
в последних 10 % — 0,438, а местами 0,00 и 0,21. При этом края дают 14 % и 8 %
выпуска. Если довести их до потолка разбивки (0,93), общая точность вырастет
примерно на 0,07 — это и есть весь резерв, который виден в данных.

Инструкции для начала и конца лежат отдельно (rules/НАЧАЛО.md, rules/КОНЕЦ.md)
и написаны по разбору пяти размеченных эфиров, с дословными примерами.

  edges.py --slot 7 --head 0.12 --tail 0.12 --mode replace
"""
import json, os, re, sys, time, argparse
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import call, render_lines, DATA, MODEL

RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")

TASK = """
=== ТВОЯ ЗАДАЧА ===

Тебе дают {which} стрима: реплики с временем, длительностью и пометками
звука. Реши по каждой реплике, войдёт она в выпуск или нет.

Думай подробно: что здесь происходит, где живое, где провисание, где
техническая возня. Опирайся на инструкцию выше — в ней разобрано, как этот
монтажёр поступает именно с {which2} выпуска.

ФОРМАТ ОТВЕТА — только строки:
номер-номер:1   — этот отрезок войдёт в выпуск
номер-номер:0   — этот отрезок вырезается

Покрой ВСЕ номера подряд, по возрастанию. Никаких пояснений."""


def parse(txt, lo, hi):
    out = {}
    for m in re.finditer(r"(\d+)\s*(?:[-–—]\s*(\d+))?\s*[:=]\s*([01])\b", txt):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        v = int(m.group(3))
        if b < a or b - a > 3000:
            continue
        for i in range(max(a, lo), min(b, hi) + 1):
            out[i] = v
    return out


def mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def run_zone(model, rules_file, which, which2, lines, mapping, lo, hi,
             chunk, reasoning, temp):
    """прогон языковой модели по зоне реплик [lo, hi]"""
    system = open(rules_file).read() + TASK.format(which=which, which2=which2)
    bounds = [(i, min(i + chunk - 1, hi)) for i in range(lo, hi + 1, chunk)]

    def one(b):
        a_, b_ = b
        sel = [s for num, s in lines if num is not None and a_ <= num <= b_]
        user = (f"{which.capitalize()} стрима, реплики {a_}–{b_}.\n\n"
                + "\n".join(sel) + f"\n\nРеши судьбу реплик {a_}–{b_}.")
        try:
            txt, u = call(model, system, user, temperature=temp,
                          max_tokens=20000, reasoning=reasoning)
        except Exception as e:
            print(f"    провал {a_}-{b_}: {e}", flush=True)
            return {}, {}
        d = parse(txt, a_, b_)
        keep = sum(1 for v in d.values() if v)
        print(f"    {a_}-{b_}: решено {len(d)}, оставлено {keep}", flush=True)
        return d, u

    dec, usage = {}, []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for d, u in ex.map(one, bounds):
            dec.update(d); usage.append(u)
    return dec, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--head", type=float, default=0.12)
    ap.add_argument("--tail", type=float, default=0.12)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--mode", default="replace", choices=["replace", "mix"],
                    help="заменить оценку модели на краях или смешать с ней")
    ap.add_argument("--w", type=float, default=0.5, help="вес языковой при mix")
    ap.add_argument("--tag", default="edges")
    a = ap.parse_args()

    blocks = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    lines, mapping = render_lines(blocks, with_audio=True, density=True)
    N = len(mapping)
    d0 = np.load(f"{DATA}/final_s{a.slot}.npy", allow_pickle=True).item()
    p, y, dur, start = d0["mix"].copy(), d0["y"], d0["dur"], d0["start"]
    T = float(dur.sum())

    # границы зон в номерах реплик
    head_last = max(n for n in mapping if start[mapping[n]] <= a.head * T)
    tail_first = min(n for n in mapping if start[mapping[n]] >= (1 - a.tail) * T)
    print(f"слот {a.slot}: начало — реплики 1–{head_last} "
          f"({a.head:.0%} эфира), конец — {tail_first}–{N}")

    t0 = time.time()
    dec, usage = {}, []
    for rules_file, which, which2, lo, hi in (
            (f"{RULES}/НАЧАЛО.md", "начало", "началом", 1, head_last),
            (f"{RULES}/КОНЕЦ.md", "конец", "концовкой", tail_first, N)):
        if not os.path.exists(rules_file):
            print(f"  нет файла {rules_file}, пропуск"); continue
        print(f"  {which}:", flush=True)
        d, u = run_zone(a.model, rules_file, which, which2, lines, mapping,
                        lo, hi, a.chunk, a.reasoning, a.temp)
        dec.update(d); usage += u

    pi = sum(u.get("prompt_tokens", 0) for u in usage)
    po = sum(u.get("completion_tokens", 0) for u in usage)
    cost = (pi * 0.07 + po * 0.13) / 1e6

    # переносим решения на ленту блоков
    llm = np.full(len(blocks), np.nan)
    for num, v in dec.items():
        if num in mapping:
            llm[mapping[num]] = v
    idx = np.flatnonzero(~np.isnan(llm))
    zone = np.zeros(len(blocks), bool)
    if len(idx):
        lo_b, hi_b = mapping[1], mapping[N]
        for num in dec:
            zone[mapping[num]] = True
        llm = np.interp(np.arange(len(blocks)), idx, llm[idx])

    def rank01(v):
        r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
        return r / max(len(v) - 1, 1)

    base = rank01(p)
    new = base.copy()
    if a.mode == "replace":
        new[zone] = llm[zone] * 0.5 + base[zone] * 0.001   # решение модели
    else:
        new[zone] = (1 - a.w) * base[zone] + a.w * llm[zone]

    rate = float((dur * y).sum() / T)

    def acc(v):
        o = np.argsort(-(v + np.random.default_rng(0).random(len(v)) * 1e-9))
        c = np.cumsum(dur[o]) / T
        k = np.zeros(len(v), bool); k[o[c <= rate]] = True
        return (dur * y)[k].sum() / dur[k].sum()

    print(f"\n{len(usage)} запросов за {time.time()-t0:.0f} с, ${cost:.4f}")
    print(f"было (только модель): {acc(base):.3f}")
    print(f"стало (края языковой моделью): {acc(new):.3f}")
    for lo, hi, nm in ((0, a.head, "начало"), (1 - a.tail, 1.0, "конец")):
        m = (start >= lo * T) & (start < hi * T)
        if m.sum() == 0:
            continue
        r2 = (dur * y)[m].sum() / dur[m].sum()

        def acc_zone(v):
            o = np.argsort(-(v[m] + np.random.default_rng(0).random(m.sum()) * 1e-9))
            c = np.cumsum(dur[m][o]) / dur[m].sum()
            k = np.zeros(m.sum(), bool); k[o[c <= r2]] = True
            return (dur[m] * y[m])[k].sum() / dur[m][k].sum()
        print(f"  {nm}: модель {acc_zone(base):.3f} -> языковая {acc_zone(llm):.3f}")

    os.makedirs(f"{DATA}/edges", exist_ok=True)
    json.dump({"slot": a.slot, "decisions": {str(k): v for k, v in dec.items()},
               "cost_usd": round(cost, 4)},
              open(f"{DATA}/edges/s{a.slot}_{a.tag}.json", "w"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
