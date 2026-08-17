#!/usr/bin/env python3
"""
Современная видео-языковая модель (Qwen3-VL-8B) как судья по кадру.

Почему повторяю попытку: первый заход был на LLaVA-1.5 образца 2023 года,
и её сигнал совпал с CLIP. Но это потолок ТОЙ модели, а не канала
изображения — вывод «видео насыщено» был сделан преждевременно.
Qwen3-VL видит действия и взаимодействия, а не только тип сцены.

Все вопросы к кадру задаются одной пачкой за проход, поэтому кадр
кодируется один раз. Ответ не генерируем — берём вероятность «Да»
против «Нет».

  vlm_qwen3.py [шаг_с] [слоты...]
"""
import os, sys, time, glob, subprocess
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

VID = os.path.join(os.environ.get("SEM_WORK", "work/window"), "vid")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TMAX = 80
FPS = 2.0

SYSTEM = ("Это кадр из уличного стрима блогера. Отвечай одним словом: "
          "Да или Нет.")

QUESTIONS = [
    ("people",  "В кадре есть другие люди, кроме самого блогера?"),
    ("interact", "Люди в кадре взаимодействуют друг с другом?"),
    ("action",  "В кадре происходит какое-то действие или событие?"),
    ("emotion", "У людей в кадре видны яркие эмоции?"),
    ("car",     "Кадр снят внутри едущей машины?"),
    ("transit", "Это переезд или переход — блогер просто добирается куда-то?"),
    ("dead",    "Кадр пустой, в нём ничего не происходит?"),
    ("bad",     "Кадр смазан, засвечен или снят в никуда?"),
    ("food",    "В кадре есть еда, напитки или застолье?"),
    ("watch",   "Такой кадр интересно смотреть в ютуб-выпуске?"),
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

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    dev = mdl.device
    tok = proc.tokenizer
    YES = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Да", "да")]
    NO = [tok(x, add_special_tokens=False).input_ids[0] for x in ("Нет", "нет")]

    def prompts(img):
        out = []
        for _, q in QUESTIONS:
            msg = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                   {"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": q}]}]
            out.append(proc.apply_chat_template(msg, tokenize=False,
                                                add_generation_prompt=True))
        return out

    for s in slots:
        p = f"{OUT}/vlm2_s{s}.npz"
        if os.path.exists(p):
            print(f"слот {s}: уже есть", flush=True); continue
        files = sorted(glob.glob(f"{VID}/s{s}/f*.jpg"))[::step]
        if not files:
            print(f"слот {s}: кадров нет", flush=True); continue
        out = np.zeros((len(files), len(QUESTIONS)), np.float32)
        t0 = time.time()
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            txts = prompts(img)
            inp = proc(text=txts, images=[img] * len(txts),
                       return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                lg = mdl(**inp).logits[:, -1].float()
            out[i] = (torch.logsumexp(lg[:, YES], 1) -
                      torch.logsumexp(lg[:, NO], 1)).cpu().numpy()
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
