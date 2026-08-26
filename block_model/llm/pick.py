#!/usr/bin/env python3
"""
Выбор окон: модель читает весь стрим сразу и называет куски для выпуска.

Замер seg_oracle.py: идеальный выбор 24 окон даёт точность 0,82, восьми
окон — 0,75. То есть цель достижима, если правильно выбрать десятки окон,
а не оценивать каждую реплику отдельно (там модель вязнет: 0,44).

Стрим целиком влезает в контекст (30–70 тыс. токенов при миллионе).
Отвечает модель диапазонами номеров реплик — числа маленькие, ошибиться
трудно, границы попадают на границу реплики.

  pick.py --slot 7 --rules rules/base.md --tag pick1
"""
import json, os, re, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import call, render_lines, DATA, MODEL

SYSTEM = """Ты — опытный монтажёр ютуб-канала с уличными IRL-стримами.

Канал: уличные IRL-стримы блогера. Он ходит и ездит по городу
с друзьями, общается со случайными людьми, попадает в разные ситуации.
Стрим идёт 2–4 часа, из него ты делаешь выпуск для YouTube.
{rules}
КАК УСТРОЕН ВЫПУСК. Это не нарезка лучших секунд, а связный фильм из
15–30 крупных кусков. Большая часть хронометража — куски длиннее минуты:
целые сцены, где что-то происходит, взятые почти целиком. Между ними
выброшены переезды, ожидание, повторы и разговоры ни о чём.

ТВОЯ ЗАДАЧА. Прочитай весь стрим и выпиши куски, которые войдут в выпуск.

Формат ответа — только строки вида:
начало-конец | краткое название куска

где начало и конец — номера реплик. Например:
145-320 | покупка формы на рынке
470-610 | первый матч, счёт растёт

Требования:
- куски идут по возрастанию номеров и не пересекаются;
- КАЖДЫЙ кусок — целая сцена: как правило 30–150 реплик, то есть несколько
  минут экрана. Куски по 5–10 реплик не годятся: это выдёргивание отдельных
  фраз, из такого выпуск не собрать;
- покрой ВЕСЬ эфир от начала до конца, а не только первую его часть:
  последняя треть стрима тоже должна быть разобрана;
- сколько всего взять — решаешь сам, исходя из того, сколько в этом эфире
  стоящего материала;
- никаких пояснений до и после списка."""

USER = """Полная расшифровка стрима, {n} реплик, эфир {mins} минут.

{text}

Выпиши куски для выпуска."""


def parse_ranges(txt, n):
    out = []
    for m in re.finditer(r"^\s*(\d+)\s*[-–—]\s*(\d+)", txt, re.M):
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= b <= n:
            out.append((a, b))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--rules", default="")
    ap.add_argument("--tag", default="pick")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reasoning", action="store_true")
    a = ap.parse_args()

    blocks = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    lines, mapping = render_lines(blocks, with_audio=True)
    text = "\n".join(s for _, s in lines)
    N = len(mapping)
    mins = blocks[-1]["end"] / 60
    rules = ("\n" + open(a.rules).read() + "\n") if a.rules else "\n"

    t0 = time.time()
    txt, u = call(a.model, SYSTEM.format(rules=rules),
                  USER.format(n=N, mins=int(mins), text=text),
                  temperature=a.temp, max_tokens=8000,
                  reasoning=a.reasoning, seed=a.seed)
    rng = parse_ranges(txt, N)
    scores = {}
    for lo, hi in rng:
        for k in range(lo, hi + 1):
            if k in mapping:
                scores[mapping[k]] = 10
    for k, bi in mapping.items():
        scores.setdefault(bi, 0)

    cost = u.get("prompt_tokens", 0) / 1e6 * 0.07 + u.get("completion_tokens", 0) / 1e6 * 0.13
    os.makedirs(f"{DATA}/ans", exist_ok=True)
    out = f"{DATA}/ans/s{a.slot}_{a.tag}.json"
    json.dump({"slot": a.slot, "tag": a.tag, "model": a.model, "mode": "pick",
               "ranges": rng, "scores": {str(k): v for k, v in scores.items()},
               "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens"),
               "cost_usd": round(cost, 4), "raw": txt}, open(out, "w"),
              ensure_ascii=False)
    print(f"{a.model}: 1 запрос за {time.time()-t0:.0f} с, "
          f"вход {u.get('prompt_tokens')} / выход {u.get('completion_tokens')}, "
          f"${cost:.4f}")
    print(f"кусков: {len(rng)}, реплик в них: {sum(b-a_+1 for a_,b in rng)}/{N}")
    print(txt[:1200])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import score as S
    blocks_ = S.load(a.slot)
    keep = [False] * len(blocks_)
    import numpy as np
    s = S.fill_pauses(blocks_, scores)
    m = S.metrics(blocks_, s >= 5)
    print(f"\nвыпуск {m['минут']:.0f} мин ({m['доля']:.1%}), у монтажёра "
          f"{m['минут у монтажёра']:.0f} мин ({m['случайно']:.1%})")
    print(f"точность {m['точность']:.3f}  полнота {m['полнота']:.3f}  "
          f"IoU {m['IoU']:.3f}")


if __name__ == "__main__":
    main()
