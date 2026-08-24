#!/usr/bin/env python3
"""
Спросить языковую модель через OpenRouter, что оставить в выпуске.

Устройство запроса выстрадано первым провалом:
  * реплики нумеруются ПОДРЯД (1..N). В ленте есть блоки-паузы, из-за них
    номера шли с дырами, и модель половину ответа тратила на разбор,
    почему пропущен номер 654;
  * рассуждение выключено. Гибридная модель уходила рассуждать по каждой
    реплике и упиралась в лимит, так и не дойдя до ответа: два куска
    из семи вернулись пустыми;
  * ответ — диапазонами «12-40:0», это коротко и заставляет мыслить
    кусками, а не отдельными репликами.

  ask.py --slot 7 --rules rules/base.md --tag base --chunk 150

Кладёт data/ans/s<слот>_<tag>.json: оценки по номерам блоков, стоимость,
сырые ответы.
"""
import json, os, re, sys, time, argparse, urllib.request, urllib.error
import concurrent.futures as cf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"

SYSTEM = """Ты — опытный монтажёр ютуб-канала с уличными IRL-стримами.

Канал: уличные IRL-стримы блогера. Он ходит и ездит по городу
с друзьями, общается со случайными людьми, попадает в разные ситуации.
Стрим идёт 2–4 часа, из него монтажёр делает выпуск для YouTube.

Твоя работа: по расшифровке стрима решить, какие куски войдут в выпуск.
{rules}
ФОРМАТ ОТВЕТА — строго. Каждой реплике поставь оценку от 0 до 10 —
насколько она достойна попасть в выпуск:
  0–1  мёртвый материал: молчание, бормотание, техническая возня
  2–3  скучно и вторично, но речь по делу
  4–5  среднее: годится, если выпуск не из чего собирать
  6–7  хорошо: живой разговор, дельная мысль, забавно
  8–10 ради этого и смотрят: смешно, неожиданно, событие, яркая эмоция

Пользуйся всей шкалой. Если у тебя половина реплик получила одну и ту же
оценку — значит ты не различаешь материал, вернись и раздели их.

Отвечай ТОЛЬКО строками «номер-номер:оценка» или «номер:оценка».
Соседние реплики с одинаковой оценкой объединяй в диапазон.
Покрой ВСЕ номера куска подряд, ничего не пропуская.
Никаких пояснений, никакого текста, никаких рассуждений — только строки.

Пример правильного ответа:
12-19:1
20-25:5
26:9
27-30:3"""

USER = """Расшифровка стрима, реплики {a}–{b} (всего в стриме {n}, это время {t0}–{t1}).

{text}

Ответь оценками для реплик {a}–{b}."""


def call(model, system, user, temperature=0.0, max_tokens=8000, retries=4,
         reasoning=False, seed=None):
    last = None
    for k in range(retries):
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # рассуждение включаем явно: без этого поля модель просто не думает
        payload["reasoning"] = ({"effort": "high"} if reasoning
                                else {"enabled": False})
        if seed is not None:
            payload["seed"] = seed
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lexeler/stream-edit-mining",
            "X-Title": "stream-edit-mining"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            txt = msg.get("content") or ""
            u = d.get("usage", {})
            if not txt.strip():
                print(f"  пустой ответ (finish={ch.get('finish_reason')}), повтор",
                      flush=True)
                last = RuntimeError("пусто"); time.sleep(2); continue
            return txt, u
        except Exception as e:
            last = e
            err = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    err = e.read().decode()[:300]
                except Exception:
                    pass
            print(f"  попытка {k+1}: {e} {err}", flush=True)
            time.sleep(4 * (k + 1))
    raise RuntimeError(f"не удалось: {last}")


def parse(txt, lo, hi):
    """строки «12-40:2» -> {локальный номер: оценка}, только в пределах куска"""
    out = {}
    for m in re.finditer(r"(\d+)\s*(?:[-–—]\s*(\d+))?\s*[:=]\s*(10|\d)\b", txt):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        v = int(m.group(3))
        if b < a or b - a > 3000:
            continue
        for i in range(max(a, lo), min(b, hi) + 1):
            out[i] = v
    return out


def mmss(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def render_lines(blocks, with_audio=True, density=False):
    """реплики с подряд идущими номерами; паузы — строкой без номера.

    density=True добавляет пометку плотности речи. Это не украшение: на 761
    контрастной границе плотность речи (знаков в секунду) предсказывает
    решение монтажёра в 74,6 % случаев, а тема — в 51,8 %, то есть никак.
    Языковая модель читает слова, но сама плотность не считает, поэтому
    показываем её явно. Пороги — из замера: <6 зн/с оставляется в 41 %
    случаев, 6–10 в 55 %, 10–14 в 60 %, свыше 14 в 65 %.
    """
    lines, mapping = [], {}
    num = 0
    dens = []
    for b in blocks:
        if b["kind"] == "speech":
            dens.append(len(b["text"]) / max(b["dur"], 0.3))
    for b in blocks:
        if b["kind"] == "pause":
            if b["dur"] >= 4:
                lines.append((None, f"      ... пауза {b['dur']:.0f} с ..."))
            continue
        num += 1
        mapping[num] = b["i"]
        marks = []
        if density:
            d = len(b["text"]) / max(b["dur"], 0.3)
            if d < 6:
                marks.append("речь рваная")
            elif d >= 14:
                marks.append("речь плотная")
        if with_audio and b.get("au"):
            marks += b["au"]
        au = " [" + ", ".join(marks) + "]" if marks else ""
        lines.append((num, f"{num}. {mmss(b['start'])} ({b['dur']:.0f}с){au} {b['text']}"))
    return lines, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--rules", default="")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overview", default="")
    ap.add_argument("--audio", action="store_true", help="показывать пометки звука")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--reasoning", action="store_true")
    a = ap.parse_args()

    blocks = json.load(open(f"{DATA}/s{a.slot}.blocks.json"))["blocks"]
    rules = ("\n" + open(a.rules).read() + "\n") if a.rules else "\n"
    system = SYSTEM.format(rules=rules)
    if a.overview and os.path.exists(a.overview):
        system += "\n\nЭТОТ СТРИМ ЦЕЛИКОМ (о чём он, по порядку):\n" \
                  + open(a.overview).read()

    lines, mapping = render_lines(blocks, a.audio)
    N = len(mapping)
    # куски по номерам реплик, с перекрытием контекста в 15 реплик
    bounds = [(i + 1, min(i + a.chunk, N)) for i in range(0, N, a.chunk)]

    def one(k):
        lo, hi = bounds[k]
        ctx_lo = max(1, lo - 15)
        txt_lines = []
        for num, s in lines:
            if num is None:
                if txt_lines:
                    txt_lines.append(s)
                continue
            if ctx_lo <= num <= hi:
                txt_lines.append(s)
        b0 = blocks[mapping[lo]]; b1 = blocks[mapping[hi]]
        user = USER.format(a=lo, b=hi, n=N, text="\n".join(txt_lines),
                           t0=mmss(b0["start"]), t1=mmss(b1["end"]))
        try:
            txt, u = call(a.model, system, user, temperature=a.temp,
                          reasoning=a.reasoning, seed=a.seed)
        except Exception as e:
            print(f"  кусок {k+1}/{len(bounds)} ({lo}-{hi}): ПРОВАЛ {e}", flush=True)
            return {"k": k, "scores": {}, "usage": {}, "raw": ""}
        sc = parse(txt, lo, hi)
        print(f"  кусок {k+1}/{len(bounds)} реплики {lo}-{hi}: "
              f"{len(sc)}/{hi-lo+1} решений, {u.get('completion_tokens',0)} токенов "
              f"выхода", flush=True)
        return {"k": k, "scores": sc, "usage": u, "raw": txt}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, range(len(bounds))))

    local = {}
    for r in res:
        local.update(r["scores"])
    scores = {mapping[k]: v for k, v in local.items() if k in mapping}
    pi = sum(r["usage"].get("prompt_tokens", 0) for r in res)
    po = sum(r["usage"].get("completion_tokens", 0) for r in res)
    cost = pi / 1e6 * 0.07 + po / 1e6 * 0.13
    os.makedirs(f"{DATA}/ans", exist_ok=True)
    out = f"{DATA}/ans/s{a.slot}_{a.tag}.json"
    json.dump({"slot": a.slot, "tag": a.tag, "model": a.model, "rules": a.rules,
               "chunk": a.chunk, "audio": a.audio, "overview": a.overview,
               "scores": {str(k): v for k, v in scores.items()},
               "tokens_in": pi, "tokens_out": po, "cost_usd": round(cost, 4),
               "raw": [r["raw"] for r in sorted(res, key=lambda z: z["k"])]},
              open(out, "w"), ensure_ascii=False)
    print(f"\n{a.model}: {len(bounds)} запросов за {time.time()-t0:.0f} с, "
          f"вход {pi} / выход {po}, ~${cost:.4f}")
    print(f"покрыто реплик: {len(local)}/{N} ({len(local)/N:.1%}) -> {out}")

    sys.path.insert(0, BASE)
    import score as S
    S.report(a.slot, scores, f"[{a.tag}]")


if __name__ == "__main__":
    main()
