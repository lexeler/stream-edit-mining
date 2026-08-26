#!/usr/bin/env python3
"""
Начала эфиров через OpenRouter: сравнение моделей на узкой задаче.

Прошлое сравнение (весь эфир, общая инструкция) дало Kimi K3 0,429,
DeepSeek V4 Flash 0,431, Gemini 3 Flash 0,449 — то есть все примерно
одинаково плохо. Здесь задача другая и гораздо более узкая: первые
20 минут эфира, где монтаж устроен шаблонно, плюс инструкция, описывающая
эту структуру. Расклад может оказаться иным.

Условия у всех моделей одинаковые: тот же файл зон, та же инструкция.
Разница только в том, кто отвечает.

  ./startllm.py --model deepseek/deepseek-v4-flash --slots 1,3,4,5,6,7
"""
import json, os, re, time, argparse, urllib.request, urllib.error

KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
ZD = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "zones")


def call(model, system, user, retries=4, reasoning=True):
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "temperature": 0.0, "max_tokens": 12000}
    if reasoning:
        payload["reasoning"] = {"effort": "high"}
    for k in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            ch = r["choices"][0]
            fin = ch.get("finish_reason")
            txt = ch["message"]["content"] or ""
            u = r.get("usage", {})
            if fin == "content_filter":
                print(f"    ВНИМАНИЕ: сработал фильтр содержимого")
            return txt, u, fin
        except Exception as e:
            if k == retries - 1:
                print(f"    ошибка: {e}")
                return "", {}, "error"
            time.sleep(3 * (k + 1))
    return "", {}, "error"


def parse(txt, need):
    """достаём {Z0000: оценка} из ответа, как бы модель его ни обернула"""
    out = {}
    m = re.search(r"\{[^{}]*\}", txt, re.S)
    if m:
        try:
            for k, v in json.loads(m.group(0)).items():
                kk = str(k).strip().lstrip("Zz")
                if kk.isdigit():
                    out[f"Z{int(kk):04d}"] = float(v)
        except Exception:
            pass
    if len(out) < need * 0.8:                      # запасной разбор по строкам
        for a, b in re.findall(r"Z?(\d{1,4})\D{1,4}(\d{1,2})(?!\d)", txt):
            i = int(a)
            if i < need:
                out.setdefault(f"Z{i:04d}", float(b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slots", default="1,3,4,5,6,7")
    ap.add_argument("--tag", default="")
    ap.add_argument("--need", type=int, default=40)
    a = ap.parse_args()
    tag = a.tag or a.model.split("/")[-1].replace(".", "")
    system = open(f"{ZD}/ИНСТРУКЦИЯ_НАЧАЛО.md").read()
    system += ("\n\nОТВЕТ: верни ТОЛЬКО JSON вида "
               '{"Z0000": 9, "Z0001": 8, ...} с ровно 40 записями Z0000–Z0039, '
               "без пояснений до и после.")
    cost = 0.0
    for s in a.slots.split(","):
        body = open(f"{ZD}/start_s{s}.txt").read()
        t0 = time.time()
        txt, u, fin = call(a.model, system,
                           "Оцени зоны Z0000–Z0039 этого куска стрима:\n\n" + body)
        d = parse(txt, a.need)
        p = f"{ZD}/startLLM_{tag}_s{s}.json"
        json.dump(d, open(p, "w"))
        print(f"  слот {s}: {len(d)}/{a.need} оценок, {time.time()-t0:.0f} с, "
              f"токенов {u.get('total_tokens', 0)}, finish={fin}")
    print(f"{a.model}: готово")


if __name__ == "__main__":
    main()
