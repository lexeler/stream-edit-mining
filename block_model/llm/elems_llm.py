#!/usr/bin/env python3
"""
Прогон ленты элементов через языковую модель, окнами по 5 минут.

Знание передаётся примерами размеченных начал из ДРУГИХ слотов, а не
описанием правил — иначе получается утечка (см. found/НАЧАЛО20.md).

Окна перекрываются наполовину, оценки по перекрытиям усредняются.

  ./elems_llm.py --model deepseek/deepseek-v4-flash --slot 7
"""
import json, os, re, time, argparse, urllib.request

KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
ED = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "elems")


def call(model, system, user, retries=4, reasoning=True, max_tokens=16000):
    payload = {"model": model,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "temperature": 0.0, "max_tokens": max_tokens}
    if reasoning:
        payload["reasoning"] = {"effort": "high"}
    for k in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            ch = r["choices"][0]
            return (ch["message"]["content"] or "", r.get("usage", {}),
                    ch.get("finish_reason"))
        except Exception as e:
            if k == retries - 1:
                return "", {}, f"error: {e}"
            time.sleep(4 * (k + 1))
    return "", {}, "error"


def parse(txt, ids):
    want = set(str(i) for i in ids)
    out = {}
    for m in re.finditer(r'"?(\d{1,5})"?\s*:\s*(\d{1,2})(?!\d)', txt):
        k, v = m.group(1), float(m.group(2))
        if k in want and v <= 10:
            out.setdefault(k, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or a.model.split("/")[-1].replace(".", "")
    system = open(f"{ED}/ИНСТРУКЦИЯ.md").read()
    ex = open(f"{ED}/examples_for_s{a.slot}.txt").read()

    pref = os.environ.get("PREF", "")
    wins = sorted(f for f in os.listdir(ED)
                  if f.startswith(f"{pref}s{a.slot}_w") and f.endswith(".txt"))
    acc, cnt = {}, {}
    t0 = time.time()
    tot_tok = 0
    for w in wins:
        ids = json.load(open(f"{ED}/{w.replace('.txt', '.ids.json')}"))
        body = open(f"{ED}/{w}").read()
        user = (ex + "\n\n=== ТЕПЕРЬ РАБОЧИЙ КУСОК ===\n"
                "Это другой эфир того же канала. Оцени элементы из раздела "
                "«ОЦЕНИТЬ ЭТИ ЭЛЕМЕНТЫ».\n\n" + body)
        txt, u, fin = call(a.model, system, user)
        d = parse(txt, ids)
        tot_tok += u.get("total_tokens", 0)
        for k, v in d.items():
            acc[k] = acc.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
        print(f"  {w}: {len(d)}/{len(ids)} оценок, finish={fin}", flush=True)
    res = {k: acc[k] / cnt[k] for k in acc}
    p = f"{ED}/pred_{tag}_s{a.slot}.json"
    json.dump(res, open(p, "w"))
    print(f"{a.model} слот {a.slot}: {len(res)} элементов оценено, "
          f"{tot_tok} токенов, {time.time()-t0:.0f} с -> {p}")


if __name__ == "__main__":
    main()
