#!/usr/bin/env python3
"""
Видео-языковая модель как судья по картинке.

Зачем: главная находка по тексту — языковая модель, отвечающая на описательные
вопросы, оказалась сильнее всего остального вместе взятого, потому что
понимает происходящее, а не узнаёт слова. С картинкой должно быть так же:
CLIP сверяет кадр с готовыми ярлыками, а LLaVA отвечает на вопрос о сцене.

Устройство то же, что у текстового судьи: кадр кодируется ОДИН раз,
вопросы переиспользуют его кэш. Ответ не генерируем — берём вероятность
«Yes» против «No».

  vlm_llava.py [шаг_в_секундах] [слоты...]
"""
import os, sys, time, glob, subprocess
import numpy as np
import torch
from PIL import Image
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          DynamicCache)

VID = os.path.join(os.environ.get("SEM_WORK", "work/window"), "vid")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "llava-hf/llava-1.5-7b-hf"
TMAX = 80
FPS = 2.0                      # кадры извлечены с такой частотой

QUESTIONS = [
    ("people",  "Are there several people in this scene?"),
    ("talk",    "Are people interacting or talking with each other?"),
    ("car",     "Is the camera inside a moving vehicle?"),
    ("action",  "Is something interesting or unusual happening here?"),
    ("boring",  "Is this an empty scene where nothing is happening?"),
    ("food",    "Are there food or drinks in this scene?"),
    ("indoor",  "Is this indoors?"),
    ("bad",     "Is the image blurry, dark or pointing at nothing?"),
]


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def main():
    hop = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    slots = sys.argv[2:] or ["1", "3", "4", "5", "6", "7", "2"]
    step = max(int(hop * FPS), 1)

    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
    mdl = LlavaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device
    tok = proc.tokenizer
    YES = [tok(x, add_special_tokens=False).input_ids[-1] for x in ("Yes", "yes")]
    NO = [tok(x, add_special_tokens=False).input_ids[-1] for x in ("No", "no")]
    suf = {k: tok(q + " ASSISTANT:", add_special_tokens=False,
                  return_tensors="pt").input_ids for k, q in QUESTIONS}

    for s in slots:
        p = f"{OUT}/vlm_s{s}.npz"
        if os.path.exists(p):
            print(f"слот {s}: уже есть", flush=True); continue
        files = sorted(glob.glob(f"{VID}/s{s}/f*.jpg"))[::step]
        if not files:
            print(f"слот {s}: кадров нет", flush=True); continue
        out = np.zeros((len(files), len(QUESTIONS)), np.float32)
        t0 = time.time()
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            inp = proc(text="USER: <image>\n", images=img, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            cache = DynamicCache()
            with torch.no_grad():
                o = mdl(**inp, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            n = cache.get_seq_length()
            for j, (k, _) in enumerate(QUESTIONS):
                ids = suf[k].to(dev)
                am = torch.ones(1, n + ids.shape[1], dtype=torch.long, device=dev)
                with torch.no_grad():
                    r = mdl(input_ids=ids, attention_mask=am,
                            past_key_values=cache, use_cache=True)
                lg = r.logits[0, -1].float()
                out[i, j] = float(torch.logsumexp(lg[YES], 0) -
                                  torch.logsumexp(lg[NO], 0))
                cache.crop(n)
            if i % 100 == 0:
                t = gpu_temp()
                print(f"  слот {s}: {i}/{len(files)}  {t} °C  "
                      f"{(time.time()-t0)/max(i,1):.2f} с/кадр", flush=True)
                while t > TMAX:
                    print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                    time.sleep(60); t = gpu_temp()
        np.savez(p, scores=out, names=[k for k, _ in QUESTIONS], hop=hop)
        print(f"слот {s}: {len(files)} кадров за {time.time()-t0:.0f} с",
              flush=True)
    print("готово")


if __name__ == "__main__":
    main()
