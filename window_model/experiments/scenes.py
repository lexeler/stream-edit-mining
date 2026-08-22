#!/usr/bin/env python3
"""
Сравнение сцен между собой — то, чего не делает ни одна наша модель.

Все прежние модели оценивают кусок сам по себе. Но монтажёр решает иначе:
он смотрит весь эфир и отбирает лучшее ИЗ ИМЕЮЩЕГОСЯ. Одна и та же сцена
войдёт в скучный стрим и вылетит из насыщенного.

Здесь языковая модель получает в один приём список ВСЕХ сцен стрима
(по минуте, короткой выжимкой расшифровки) и отвечает про каждую на фоне
остальных. Список кодируется один раз, вопросы переиспользуют его кэш —
поэтому весь стрим обходится в несколько минут.

  scenes.py [длина_сцены_с] [слоты...]
"""
import json, collections, os, sys, time, subprocess
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SCENE = 60.0
CHARS = 200
TMAX = 80

SYSTEM = (
    "Ты помощник монтажёра ютуб-канала с уличными IRL-стримами. Это уличные стримы "
    "по три часа: блогер ходит по городу, ездит, общается с людьми. "
    "Из стрима делают выпуск примерно на час — берут лучшее, "
    "остальное выбрасывают."
)

QUESTIONS = [
    ("better", "Сцена {k} интереснее среднего по этому стриму?"),
    ("cut",    "Сцену {k} стоит выбросить как скучную?"),
    ("worth",  "В сцене {k} происходит что-то стоящее внимания?"),
    ("top",    "Сцена {k} войдёт в лучшую треть этого стрима?"),
]


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def scene_texts(rows, scene, chars):
    words = [(r["start"], r["word"]) for r in rows if r["type"] == "word"]
    ws = np.array([w[0] for w in words]) if words else np.zeros(1)
    T = max(r["end"] for r in rows)
    n = int(np.ceil(T / scene))
    out = []
    for k in range(n):
        lo = int(np.searchsorted(ws, k * scene))
        hi = int(np.searchsorted(ws, (k + 1) * scene))
        t = " ".join(w[1] for w in words[lo:hi]).strip()
        out.append(t[:chars] if t else "(тишина)")
    return out


def main():
    scene = float(sys.argv[1]) if len(sys.argv) > 1 else SCENE
    slots = sys.argv[2:] or ["1", "3", "4", "5", "6", "7", "2"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device
    YES = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Да", "да")]
    NO = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Нет", "нет")]

    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["slot"]].append(r)
        for s in sorted(by):
            if s not in slots:
                continue
            p = f"{DATA}/scene_s{s}.npz"
            if os.path.exists(p):
                print(f"слот {s}: уже есть", flush=True); continue
            # память карты освобождаем ДО построения нового кэша: иначе
            # старый список сцен и новый живут одновременно и не влезают
            torch.cuda.empty_cache()
            txt = scene_texts(by[s], scene, CHARS)
            lst = "\n".join(f"{k+1}. {t}" for k, t in enumerate(txt))
            head = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                    "<|im_start|>user\nВот весь стрим по минутам, "
                    f"всего {len(txt)} сцен:\n\n" + lst + "\n\n")
            pre = tok(head, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(dev)
            t0 = time.time()
            # список сцен длинный (до 20 тысяч токенов), разом не влезает
            # в память — подаём частями, накапливая кэш
            cache = DynamicCache()
            CH = 1024
            for a in range(0, pre.shape[1], CH):
                part = pre[:, a:a + CH]
                am = torch.ones(1, a + part.shape[1], dtype=torch.long, device=dev)
                with torch.no_grad():
                    o = mdl.model(part, attention_mask=am,
                                  past_key_values=cache, use_cache=True)
                cache = o.past_key_values
            n0 = cache.get_seq_length()
            print(f"слот {s}: {len(txt)} сцен, {n0} токенов в списке, "
                  f"{time.time()-t0:.0f} с", flush=True)

            out = np.zeros((len(txt), len(QUESTIONS)), np.float32)
            for k in range(len(txt)):
                for j, (name, q) in enumerate(QUESTIONS):
                    suf = tok(q.format(k=k + 1) + " Ответь одним словом."
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
                if k % 50 == 0:
                    t = gpu_temp()
                    print(f"  сцена {k}/{len(txt)}  {t} °C  "
                          f"{(time.time()-t0)/max(k,1):.2f} с/сцена", flush=True)
                    while t > TMAX:
                        print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                        time.sleep(60); t = gpu_temp()
            np.savez(p, scores=out, names=[k for k, _ in QUESTIONS],
                     scene=scene)
            print(f"слот {s}: готово за {time.time()-t0:.0f} с", flush=True)
            del cache, o
            torch.cuda.empty_cache()
    print("готово")


if __name__ == "__main__":
    main()
