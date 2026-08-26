#!/usr/bin/env python3
"""
Зоны стрима по описаниям кадров: где герои и чем заняты.

Разбор слота 4: 89 % выпуска взято из одного окна «внутри музея», и тот же
возглас внутри аттракциона остаётся в 67 % случаев, а на улице между делом
в 4 %. Значит нужен признак «мы сейчас внутри занятия».

Текст для этого не годится — проверено контрольным опытом: человек, читающий
расшифровку, даёт 0,46–0,49 против 0,51–0,53 у модели. А вот кадры говорят
прямо: «магазин, примеряют форму», «поле, играют в футбол», «салон машины».

Даём модели подряд все описания кадров (одна строка на 12 секунд) и просим
разбить эфир на зоны.

  zones.py --slot 7
Кладёт data/zones/s<N>.json — список {t0, t1, тип, название}.
"""
import json, os, re, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import call, MODEL

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SEC = 12.0

SYSTEM = """Ты разбираешь уличный стрим по кадрам. Тебе дают описания кадров
подряд, по одному раз в 12 секунд, с временем.

Раздели эфир на ЗОНЫ. Зона — это отрезок, где герои находятся в одном месте
и заняты одним делом. Смена места или занятия начинает новую зону.

Каждой зоне поставь тип:
  ЗАНЯТИЕ — герои чем-то заняты в конкретном месте: играют, лазают,
            катаются, выбирают товар, участвуют в чужом деле, что-то
            пробуют. Главное: есть место и есть дело.
  ДОРОГА  — едут в машине, идут между местами.
  БЫТ     — сидят за столом, едят, отдыхают, разговаривают без дела.
  ВОЗНЯ   — ждут, ищут, собираются, настраивают камеру, стоят на месте.

ФОРМАТ ОТВЕТА — только строки:
время_начала-время_конца | ТИП | короткое название

Время в минутах и секундах, как в исходных данных. Например:
0:00-4:12 | ВОЗНЯ | сборы у дома
4:12-18:30 | ДОРОГА | едут в город
18:30-46:05 | ЗАНЯТИЕ | магазин, выбирают форму

Зоны идут подряд, без разрывов и пересечений, покрывают весь эфир.
Ожидаю 15–40 зон. Никаких пояснений."""


def mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def parse_time(s):
    m = re.match(r"(\d+):(\d+)", s.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    d = json.load(open(f"{DATA}/desc/desc_s{a.slot}.json"))
    keys = sorted(d, key=lambda k: int(re.findall(r"\d+", k)[0]))
    lines = [f"{mmss(i * SEC)} {d[k]}" for i, k in enumerate(keys)]
    text = "\n".join(lines)
    print(f"слот {a.slot}: {len(keys)} кадров, {len(text)//1000} тыс. символов")

    txt, u = call(a.model, SYSTEM, "Описания кадров стрима:\n\n" + text,
                  temperature=0.2, max_tokens=6000, reasoning=False)
    zones = []
    for ln in txt.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) < 2 or "-" not in p[0]:
            continue
        a_, b_ = p[0].split("-")[:2]
        t0, t1 = parse_time(a_), parse_time(b_)
        if t0 is None or t1 is None or t1 <= t0:
            continue
        zones.append({"t0": t0, "t1": t1, "type": p[1],
                      "name": p[2] if len(p) > 2 else ""})
    os.makedirs(f"{DATA}/zones", exist_ok=True)
    json.dump({"slot": a.slot, "zones": zones, "raw": txt},
              open(f"{DATA}/zones/s{a.slot}.json", "w"), ensure_ascii=False)
    cost = (u.get("prompt_tokens", 0) * 0.07 + u.get("completion_tokens", 0)
            * 0.13) / 1e6
    print(f"зон: {len(zones)}, ${cost:.4f}")
    for z in zones:
        print(f"  {mmss(z['t0'])}-{mmss(z['t1'])} {z['type']:>9} {z['name']}")


if __name__ == "__main__":
    main()
