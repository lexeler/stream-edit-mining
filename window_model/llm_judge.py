#!/usr/bin/env python3
"""
Языковая модель как судья, вторая попытка.

Зачем: векторы e5 отвечают на вопрос «о чём говорят», а нужен ответ на «стоит
ли это смотреть». Обучающих решений всего 2400, поэтому знание о мире выгоднее
брать готовым из языковой модели, чем выучивать.

Устройство: расшифровка окна кодируется ОДИН раз, пять вопросов
переиспользуют её кэш. Без этого было бы три часа вместо сорока минут.
Ответ не генерируем — берём вероятность слова «да» против «нет»,
это устойчивее и даёт плавную оценку вместо двоичной.
"""
import json, collections, os, sys, time, subprocess
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
HOP, CTX = 15.0, 45.0          # шаг сетки и радиус контекста, с
TMAX = 80

SYSTEM = (
    "Ты помогаешь монтажёру ютуб-канала с уличными IRL-стримами. Это уличные стримы "
    "по три часа: блогер ходит по городу, ездит, общается с людьми, "
    "попадает в разные ситуации. Из стрима монтажёр делает выпуск примерно "
    "на час — оставляет живое и смешное, разговоры с людьми, события "
    "и яркие моменты. Выбрасывает молчание, дорогу, ожидание, технические "
    "заминки и пустой трёп ни о чём. Отвечай одним словом."
)

QUESTIONS = [
    # Прямой вопрос «оставит ли монтажёр» проверен и НЕ работает: AUC 0,471,
    # хуже монетки. Языковая модель не знает вкусов монтажёра. Зато
    # описательный вопрос «происходит ли действие» дал AUC 0,767 одним
    # признаком без обучения. Поэтому спрашиваем только про то, что видно
    # в тексте, а связывать это с решением монтажёра — работа модели.
    ("event", "Происходит ли здесь какое-то действие или событие?"),
    ("people", "Разговаривает ли он с другими людьми, а не сам с собой?"),
    ("dull",  "Он молчит или говорит ни о чём?"),
    ("fun",   "Кто-то здесь шутит или смеётся?"),
    ("road",  "Он сейчас в дороге — едет, идёт, куда-то добирается?"),
    ("emo",   "Есть ли сильные эмоции — крик, восторг, ругань, спор?"),
    ("tech",  "Это техническая заминка: связь, звук, камера, зарядка?"),
    ("story", "Он рассказывает историю или объясняет что-то?"),
    ("chat",  "Он читает чат или отвечает зрителям?"),
    ("hello", "Это приветствие в начале или прощание в конце стрима?"),
]


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def texts_for(rows, hop, ctx):
    words = [(r["start"], r["word"]) for r in rows if r["type"] == "word"]
    ws = np.array([w[0] for w in words]) if words else np.zeros(1)
    T = max(r["end"] for r in rows)
    grid = np.arange(0, T + hop, hop)
    out = []
    for t in grid:
        lo = int(np.searchsorted(ws, t - ctx))
        hi = int(np.searchsorted(ws, t + ctx))
        out.append(" ".join(w[1] for w in words[lo:hi])[:1400])
    return grid, out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device if hasattr(mdl, "device") else "cuda:0"

    def ids(s):
        return tok(s, add_special_tokens=False, return_tensors="pt").input_ids

    YES = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Да", "да")]
    NO = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Нет", "нет")]
    suf = {k: ids(q + "<|im_end|>\n<|im_start|>assistant\n")
           for k, q in QUESTIONS}
    head = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"

    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["slot"]].append(r)
        for s in sorted(by):
            p = f"{DATA}/llm_s{s}.npz"
            if os.path.exists(p):
                print(f"слот {s}: уже есть", flush=True); continue
            grid, txts = texts_for(by[s], HOP, CTX)
            out = np.zeros((len(txts), len(QUESTIONS)), np.float32)
            emb = np.zeros((len(txts), mdl.config.hidden_size), np.float16)
            t0 = time.time()
            for i, tx in enumerate(txts):
                if not tx.strip():
                    out[i] = 0.0; continue
                pre = ids(head + "Кусок расшифровки стрима:\n«" + tx + "»\n\n").to(dev)
                cache = DynamicCache()
                with torch.no_grad():
                    # только основа модели, без головы: вероятности для всех
                    # позиций заняли бы 400 МБ на окно, а нужна лишь последняя
                    o = mdl.model(pre, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                emb[i] = o.last_hidden_state[0].mean(0).float().cpu().numpy()
                n = pre.shape[1]
                for j, (k, _) in enumerate(QUESTIONS):
                    with torch.no_grad():
                        r = mdl(suf[k].to(dev), past_key_values=cache, use_cache=True)
                    lg = r.logits[0, -1].float()
                    y = torch.logsumexp(lg[YES], 0)
                    nn = torch.logsumexp(lg[NO], 0)
                    out[i, j] = float(y - nn)   # логит: больше разрешения на краях
                    cache.crop(n)
                if i % 200 == 0:
                    t = gpu_temp()
                    el = time.time() - t0
                    print(f"  слот {s}: {i}/{len(txts)}  {t} °C  "
                          f"{el/max(i,1):.2f} с/окно", flush=True)
                    while t > TMAX:
                        print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                        time.sleep(60); t = gpu_temp()
            np.savez(p, grid=grid, scores=out, emb=emb,
                     names=[k for k, _ in QUESTIONS], hop=HOP)
            print(f"слот {s}: {len(txts)} окон за {time.time()-t0:.0f} с", flush=True)
    print("готово")


if __name__ == "__main__":
    main()
