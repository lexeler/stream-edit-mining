#!/usr/bin/env python3
"""
Показать модели примеры вместо того, чтобы спрашивать абстрактно.

Все прежние попытки спрашивали языковую модель о свойствах куска
(«интересно?», «событие?», «подводка?») и упирались в AUC 0,5–0,6.
Причина в том, что модель не знает вкуса ЭТОГО монтажёра. Прямой вопрос
«оставит ли он» проверен и дал 0,471 — хуже монетки.

Здесь другой ход: кладём в подсказку два-три десятка настоящих примеров
из ДРУГИХ стримов с ответом «оставил» или «выбросил», и просим оценить
новый кусок. Это перенос вкуса без дообучения.

Утечки нет: примеры для слота берутся только из остальных слотов,
для тестового слота 2 — только из обучающих шести.
"""
import json, collections, os, sys, time, subprocess
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHUNK = 30.0
NEX = 40            # примеров каждого вида
EXLEN = 400         # длина примера в символах
TMAX = 80
TRAIN = ["1", "3", "4", "5", "6", "7"]

SYSTEM = ("Ты монтажёр ютуб-канала с уличными IRL-стримами. Уличный стрим на три часа "
          "режут в выпуск на час. Ниже примеры того, что этот монтажёр "
          "оставил, и что выбросил. Отвечай одним словом: Да или Нет.")


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def chunks_of(rows):
    w = [(r["start"], r["word"]) for r in rows if r["type"] == "word"]
    ws = np.array([x[0] for x in w]) if w else np.zeros(1)
    dur = np.array([r["dur"] for r in rows], np.float32)
    st = np.array([r["start"] for r in rows], np.float32)
    fr = np.array([r.get("used_frac", 0.0) for r in rows], np.float32)  # у слота 2 меток нет и не нужно
    T = float(st[-1] + dur[-1])
    n = int(np.ceil(T / CHUNK))
    txt, lab = [], np.zeros(n, np.float32)
    ci = np.clip(((st + dur / 2) / CHUNK).astype(int), 0, n - 1)
    num = np.zeros(n); den = np.zeros(n)
    np.add.at(num, ci, fr * dur); np.add.at(den, ci, dur)
    lab = np.divide(num, np.maximum(den, 1e-9))
    for k in range(n):
        lo = int(np.searchsorted(ws, k * CHUNK))
        hi = int(np.searchsorted(ws, (k + 1) * CHUNK))
        t = " ".join(x[1] for x in w[lo:hi]).strip()
        txt.append(t[:EXLEN] if t else "(тишина)")
    return txt, lab


def main():
    slots = sys.argv[1:] or ["1", "3", "4", "5", "6", "7", "2"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device
    YES = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Да", "да")]
    NO = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Нет", "нет")]

    allrows = collections.defaultdict(list)
    for split in ("train", "test"):
        for l in open(f"{DATA}/{split}.jsonl"):
            r = json.loads(l)
            allrows[r["slot"]].append(r)
    CH = {s: chunks_of(allrows[s]) for s in sorted(allrows)}
    rng = np.random.default_rng(0)

    for s in slots:
        p = f"{DATA}/icl2_s{s}.npz"
        if os.path.exists(p):
            print(f"слот {s}: уже есть", flush=True); continue
        src = [x for x in TRAIN if x != s]
        pos, neg = [], []
        for o in src:
            txt, lab = CH[o]
            good = [i for i in range(len(txt)) if lab[i] > 0.95
                    and len(txt[i]) > 90]
            bad = [i for i in range(len(txt)) if lab[i] < 0.05
                   and len(txt[i]) > 90]
            for i in rng.choice(good, min(NEX // len(src) + 1, len(good)),
                                replace=False):
                pos.append(txt[i])
            for i in rng.choice(bad, min(NEX // len(src) + 1, len(bad)),
                                replace=False):
                neg.append(txt[i])
        pos, neg = pos[:NEX], neg[:NEX]
        ex = []
        for a, b in zip(pos, neg):
            ex.append(f"ОСТАВИЛ: «{a}»")
            ex.append(f"ВЫБРОСИЛ: «{b}»")
        head = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                "<|im_start|>user\nПримеры:\n\n" + "\n".join(ex) + "\n\n")
        pre = tok(head, add_special_tokens=False,
                  return_tensors="pt").input_ids.to(dev)
        cache = DynamicCache()
        with torch.no_grad():
            o = mdl.model(pre, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        n0 = cache.get_seq_length()
        txt, lab = CH[s]
        out = np.zeros(len(txt), np.float32)
        t0 = time.time()
        print(f"слот {s}: примеров {len(ex)}, {n0} токенов, кусков {len(txt)}",
              flush=True)
        for k, t in enumerate(txt):
            suf = tok(f"Новый кусок: «{t}»\nОставит ли монтажёр этот кусок?"
                      "<|im_end|>\n<|im_start|>assistant\n",
                      add_special_tokens=False,
                      return_tensors="pt").input_ids.to(dev)
            am = torch.ones(1, n0 + suf.shape[1], dtype=torch.long, device=dev)
            with torch.no_grad():
                r = mdl(input_ids=suf, attention_mask=am,
                        past_key_values=cache, use_cache=True)
            lg = r.logits[0, -1].float()
            out[k] = float(torch.logsumexp(lg[YES], 0) -
                           torch.logsumexp(lg[NO], 0))
            cache.crop(n0)
            if k % 500 == 0:
                tt = gpu_temp()
                print(f"  {k}/{len(txt)}  {tt} °C  "
                      f"{(time.time()-t0)/max(k,1):.3f} с/кусок", flush=True)
                while tt > TMAX:
                    print(f"  ГОРЯЧО {tt} °C — пауза 60 с", flush=True)
                    time.sleep(60); tt = gpu_temp()
        np.savez(p, scores=out[:, None], names=["icl"], hop=CHUNK)
        print(f"слот {s}: готово за {time.time()-t0:.0f} с", flush=True)
        del cache, o
        torch.cuda.empty_cache()
    print("готово")


if __name__ == "__main__":
    main()
