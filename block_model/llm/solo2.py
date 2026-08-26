#!/usr/bin/env python3
"""
Монтаж языковой моделью по кускам, с рассуждением: она сама решает, что взять.

Один запрос на весь стрим не работает: модель не удерживает две тысячи реплик,
берёт половину эфира подряд и даёт 0,362 (случайно 0,303). Поэтому эфир идёт
кусками по 150–200 реплик, но каждому куску даётся полное описание работы
монтажёра и включено рассуждение — пусть думает столько, сколько нужно.

Решение бинарное: оставить или выбросить. Никаких порогов с моей стороны,
никакой сборки — сколько модель наметила, столько и вошло, длину выпуска
она выбирает сама.

  solo2.py --slot 7 --tag c1 --chunk 150 --reasoning
"""
import json, os, re, sys, time, argparse
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import call, render_lines, DATA, MODEL
from solo import SYSTEM as FULL_RULES

SYSTEM = FULL_RULES.split("=== ТВОЯ ЗАДАЧА ===")[0] + """=== ТВОЯ ЗАДАЧА ===

Тебе дают кусок расшифровки. Реши по каждой реплике: войдёт она в выпуск
или нет.

Думай так, как думает монтажёр за монитором: что здесь происходит, чем это
место ценно, где живая сцена, а где провисание, не было ли этого уже раньше.
Внутри живой сцены не оставляй всё подряд — подчищай провисания, мычание
и лишние паузы, иначе выпуск будет вялым.

БУДЬ ЖЁСТЧЕ, ЧЕМ ХОЧЕТСЯ. Этот монтажёр выбрасывает от 55 до 75 процентов
эфира. Материал, который кажется «неплохим», он режет: в выпуск попадает
только то, что держит внимание постороннего зрителя. Если в куске у тебя
получилось оставить больше половины реплик — вернись и перепроверь, скорее
всего ты пожалел то, что он бы выбросил.

ФОРМАТ ОТВЕТА — только строки:
номер-номер:1   — этот отрезок реплик войдёт в выпуск
номер-номер:0   — этот отрезок вырезается

Покрой ВСЕ номера куска подряд, ничего не пропуская, по возрастанию.
Никаких пояснений и заголовков — только строки.

Пример:
412-419:0
420-425:1
426:0
427-431:1"""

USER = """Кусок расшифровки стрима: реплики {a}–{b} из {n}, время {t0}–{t1}.
{ctx}
{text}

Реши судьбу реплик {a}–{b}."""


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--overview", default="", help="карта стрима в контекст")
    ap.add_argument("--example", action="store_true",
                    help="показать образец настоящей разметки монтажёра")
    a = ap.parse_args()

    system = SYSTEM
    if a.example:
        # образец из ДРУГОГО эфира: как этот монтажёр решил на живом куске.
        # Правила словами не передают его манеру целиком, а пример передаёт
        ex = open(f"{DATA}/example_s5.txt").read()
        ans = open(f"{DATA}/example_s5_ans.txt").read()
        system += ("\n\n=== ОБРАЗЕЦ ЕГО РАБОТЫ ===\nВот кусок другого эфира "
                   "и то, как он сам его разметил.\n\n" + ex
                   + "\n\nЕго решение:\n" + ans
                   + "\n\nПосмотри, как часто он режет и что именно "
                     "выбрасывает внутри живых сцен.")

    blocks = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    lines, mapping = render_lines(blocks, with_audio=True, density=True)
    N = len(mapping)
    bounds = [(i + 1, min(i + a.chunk, N)) for i in range(0, N, a.chunk)]
    ctx = ""
    if a.overview and os.path.exists(a.overview):
        ctx = ("\nО ЧЁМ ЭТОТ СТРИМ ЦЕЛИКОМ (чтобы ты видел повторы и общий "
               "ход):\n" + open(a.overview).read() + "\n")

    def one(k):
        lo, hi = bounds[k]
        pre = max(1, lo - 12)
        sel = []
        for num, s in lines:
            if num is None:
                if sel:
                    sel.append(s)
                continue
            if pre <= num <= hi:
                sel.append(s)
        b0 = blocks[mapping[lo]]; b1 = blocks[mapping[hi]]
        user = USER.format(a=lo, b=hi, n=N, ctx=ctx, text="\n".join(sel),
                           t0=mmss(b0["start"]), t1=mmss(b1["end"]))
        try:
            txt, u = call(a.model, system, user, temperature=a.temp,
                          max_tokens=20000, reasoning=a.reasoning)
        except Exception as e:
            print(f"  кусок {k+1}: ПРОВАЛ {e}", flush=True)
            return {"k": k, "d": {}, "u": {}}
        d = parse(txt, lo, hi)
        keep = sum(1 for v in d.values() if v)
        print(f"  кусок {k+1}/{len(bounds)} ({lo}-{hi}): решено {len(d)}, "
              f"оставлено {keep} ({keep/max(len(d),1):.0%})", flush=True)
        return {"k": k, "d": d, "u": u}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, range(len(bounds))))
    dec = {}
    for r in res:
        dec.update(r["d"])
    pi = sum(r["u"].get("prompt_tokens", 0) for r in res)
    po = sum(r["u"].get("completion_tokens", 0) for r in res)
    # цена зависит от модели: у дешёвой 0,07/0,13 за миллион, у сильной 5/25
    PRICES = {"glm-5.2": (0.46, 1.45), "kimi-k3": (3.0, 15.0),
              "gemini-3-flash": (0.5, 3.0), "qwen3-235b": (0.09, 0.55),
              "gpt-5": (1.25, 10.0), "grok-4": (0.2, 0.5)}
    rate_in, rate_out = next(((i, o) for k, (i, o) in PRICES.items()
                              if k in a.model), (0.07, 0.13))
    cost = (pi * rate_in + po * rate_out) / 1e6

    dur = np.array([b["dur"] for b in blocks])
    y = np.array([b["used_frac"] for b in blocks])
    kind = np.array([b["kind"] for b in blocks])
    sel = np.zeros(len(blocks), bool)
    for num, v in dec.items():
        if v and num in mapping:
            sel[mapping[num]] = True
    # пауза входит в выпуск, только если обе соседние реплики взяты:
    # так решает и монтажёр — тишина между выброшенными у него не выживает
    for i in range(1, len(blocks) - 1):
        if kind[i] == "pause":
            sel[i] = sel[i - 1] and sel[i + 1]
    segs = []
    for i, b in enumerate(blocks):
        if not sel[i]:
            continue
        s, e = float(b["start"]), float(b["end"])
        if segs and s - segs[-1][1] < 1e-6:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    ov = float((dur * y)[sel].sum()); pred = float(dur[sel].sum())
    true = float((dur * y).sum())
    L = np.array([e - s for s, e in segs]) if segs else np.array([0.0])
    os.makedirs(f"{DATA}/solo", exist_ok=True)
    json.dump({"slot": a.slot, "model": a.model, "reasoning": a.reasoning,
               "decisions": {str(k): v for k, v in dec.items()},
               "segments": [[round(s, 2), round(e, 2)] for s, e in segs],
               "cost_usd": round(cost, 4)},
              open(f"{DATA}/solo/s{a.slot}_{a.tag}.json", "w"), ensure_ascii=False)
    print(f"\n{len(bounds)} запросов за {time.time()-t0:.0f} с, "
          f"вход {pi} / выход {po}, ${cost:.4f}, покрыто {len(dec)}/{N}")
    print(f"кусков {len(segs)}, выпуск {pred/60:.1f} мин ({pred/dur.sum():.1%}), "
          f"медиана {np.median(L):.1f} с, макс {L.max():.0f} с")
    print(f"у монтажёра {true/60:.1f} мин, {len(json.load(open(f'$SEM_STREAMS/{a.slot}/narezki/timecodes.json'))['segments'])} кусков")
    print(f"ТОЧНОСТЬ {ov/pred if pred else 0:.3f}  полнота {ov/true:.3f}  "
          f"случайно {true/dur.sum():.3f}")


if __name__ == "__main__":
    main()
