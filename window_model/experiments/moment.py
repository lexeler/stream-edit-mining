#!/usr/bin/env python3
"""
Подводка или сам момент — внутри одной сцены.

Наблюдение из данных. В слоте 5 сцена с танком: монтажёр ОСТАВИЛ
«Вот он контент, который мы заслужили. Ебать, дуло там! Мы в фильме, чел»
и ВЫБРОСИЛ «Пошли залазить. А чё не говорили, команды нет? Что, я готов?».
Одна сцена, одна тема, одна громкость. Разница в том, что первое — сам
момент, а второе — организационная возня перед ним.

Ни один прежний признак этого не ловит: по окну без контекста непонятно,
подводка это или кульминация, а по сцене целиком решение слишком грубое
(проверено: отбор сценами хуже, 0,553 против 0,583).

Здесь модель видит сцену на три минуты, разбитую по 15 секунд, и про
каждый кусок отвечает на фоне соседних. Сцена кодируется один раз.
"""
import json, collections, os, sys, time, subprocess
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCENE, CHUNK = 180.0, 15.0
TMAX = 80

SYSTEM = (
    "Ты помогаешь монтажёру ютуб-канала с уличными IRL-стримами. Уличный стрим на три "
    "часа режут в выпуск на час: оставляют сами моменты — реакции, шутки, "
    "события, яркие реплики. Выбрасывают подводку и возню вокруг: "
    "«пошли», «сейчас сделаем», «а как это включается», повторы и уточнения. "
    "Отвечай одним словом: Да или Нет."
)

QUESTIONS = [
    ("moment", "Кусок {k} — это сам момент, а не подводка к нему?"),
    ("setup",  "Кусок {k} — это организационная возня и уточнения?"),
    ("memory", "В куске {k} есть то, что зритель запомнит?"),
    ("skip",   "Кусок {k} можно выбросить, ничего не потеряв?"),
]


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def main():
    slots = sys.argv[1:] or ["1", "3", "4", "5", "6", "7", "2"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device
    YES = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Да", "да")]
    NO = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Нет", "нет")]
    per = int(SCENE / CHUNK)

    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["slot"]].append(r)
        for s in sorted(by):
            if s not in slots:
                continue
            p = f"{DATA}/moment_s{s}.npz"
            if os.path.exists(p):
                print(f"слот {s}: уже есть", flush=True); continue
            w = [(r["start"], r["word"]) for r in by[s] if r["type"] == "word"]
            ws = np.array([x[0] for x in w]) if w else np.zeros(1)
            T = max(r["end"] for r in by[s])
            nch = int(np.ceil(T / CHUNK))
            out = np.zeros((nch, len(QUESTIONS)), np.float32)
            t0 = time.time()
            for sc0 in range(0, nch, per):
                sc1 = min(sc0 + per, nch)
                txt = []
                for k in range(sc0, sc1):
                    lo = int(np.searchsorted(ws, k * CHUNK))
                    hi = int(np.searchsorted(ws, (k + 1) * CHUNK))
                    t = " ".join(x[1] for x in w[lo:hi]).strip()
                    txt.append(f"{k - sc0 + 1}. {t[:220] if t else '(тишина)'}")
                head = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                        "<|im_start|>user\nКусок стрима на три минуты, "
                        "по 15 секунд:\n\n" + "\n".join(txt) + "\n\n")
                pre = tok(head, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(dev)
                cache = DynamicCache()
                with torch.no_grad():
                    o = mdl.model(pre, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                n0 = cache.get_seq_length()
                for k in range(sc0, sc1):
                    for j, (_, q) in enumerate(QUESTIONS):
                        suf = tok(q.format(k=k - sc0 + 1) +
                                  "<|im_end|>\n<|im_start|>assistant\n",
                                  add_special_tokens=False,
                                  return_tensors="pt").input_ids.to(dev)
                        am = torch.ones(1, n0 + suf.shape[1], dtype=torch.long,
                                        device=dev)
                        with torch.no_grad():
                            r = mdl(input_ids=suf, attention_mask=am,
                                    past_key_values=cache, use_cache=True)
                        lg = r.logits[0, -1].float()
                        out[k, j] = float(torch.logsumexp(lg[YES], 0) -
                                          torch.logsumexp(lg[NO], 0))
                        cache.crop(n0)
                del cache, o
                if (sc0 // per) % 20 == 0:
                    t = gpu_temp()
                    print(f"  слот {s}: {sc0}/{nch}  {t} °C  "
                          f"{(time.time()-t0)/max(sc0,1):.2f} с/кусок", flush=True)
                    torch.cuda.empty_cache()
                    while t > TMAX:
                        print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                        time.sleep(60); t = gpu_temp()
            np.savez(p, scores=out, names=[k for k, _ in QUESTIONS], hop=CHUNK)
            print(f"слот {s}: {nch} кусков за {time.time()-t0:.0f} с", flush=True)
    print("готово")


if __name__ == "__main__":
    main()
