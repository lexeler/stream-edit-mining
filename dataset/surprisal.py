#!/usr/bin/env python3
"""
Неожиданность речи: насколько предсказуемо то, что он говорит.

Чего не хватало. Векторы e5 отвечают «о чём речь», вопросы к языковой модели —
«что происходит». Но ни один признак не отвечает на вопрос «а это вообще
содержательно или это словесная вода». Слова-паразиты, повторы и мычание
предсказываются моделью почти идеально, а живая мысль или неожиданный
поворот дают всплеск неожиданности.

Считаем логарифм вероятности каждого слова под языковой моделью,
переносим на время и усредняем по секундам.
"""
import json, collections, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CH, OV = 2048, 256             # длина куска и перекрытие, в токенах
GRID = 1.0


@torch.no_grad()
def main():
    slots = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device

    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["slot"]].append(r)
        for s in sorted(by):
            if s not in slots:
                continue
            p = f"{DATA}/surp_s{s}.npz"
            if os.path.exists(p):
                print(f"слот {s}: уже есть", flush=True); continue
            w = [(r["start"], r["end"], r["word"]) for r in by[s]
                 if r["type"] == "word"]
            if not w:
                continue
            t0 = time.time()
            # склеиваем текст и запоминаем, какому слову принадлежит символ
            text, owner = [], []
            for k, (a, b, x) in enumerate(w):
                if text:
                    text.append(" "); owner.append(k)
                text.append(x); owner += [k] * len(x)
            text = "".join(text)
            owner = np.array(owner)
            enc = tok(text, return_offsets_mapping=True,
                      add_special_tokens=False)
            ids = np.array(enc["input_ids"])
            off = np.array(enc["offset_mapping"])
            lp = np.full(len(ids), np.nan, np.float32)
            for a in range(0, len(ids), CH - OV):
                b = min(a + CH, len(ids))
                chunk = torch.tensor(ids[a:b])[None].to(dev)
                logits = mdl(chunk).logits[0].float()
                ls = torch.log_softmax(logits[:-1], -1)
                got = ls[torch.arange(b - a - 1), chunk[0, 1:]].cpu().numpy()
                st = a + 1 if a == 0 else a + OV
                lp[st:b] = got[st - a - 1:]
                if b == len(ids):
                    break
            # переносим на слова, потом на секунды
            T = w[-1][1]
            n = int(np.ceil(T / GRID))
            acc = np.zeros((n, 3), np.float32); cnt = np.zeros(n)
            for j in range(len(ids)):
                if np.isnan(lp[j]):
                    continue
                c = int(off[j][0])
                if c >= len(owner):
                    continue
                k = owner[c]
                t = int((w[k][0] + w[k][1]) / 2 / GRID)
                if 0 <= t < n:
                    acc[t, 0] += -lp[j]
                    acc[t, 1] = max(acc[t, 1], -lp[j])
                    acc[t, 2] += float(-lp[j] > 5.0)
                    cnt[t] += 1
            good = cnt > 0
            acc[good, 0] /= cnt[good]
            acc[good, 2] /= cnt[good]
            # пустые секунды заполняем предыдущими
            last = np.zeros(3, np.float32)
            for i in range(n):
                if good[i]:
                    last = acc[i]
                else:
                    acc[i] = last
            np.savez(p, feats=acc, grid=GRID,
                     names=["surp_mean", "surp_max", "surp_rate"])
            print(f"слот {s}: {n} секунд, {len(ids)} токенов за "
                  f"{time.time()-t0:.0f} с", flush=True)
    print("готово")


if __name__ == "__main__":
    main()
