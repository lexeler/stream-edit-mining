#!/usr/bin/env python3
"""
Описание кадров словами — то, чего нет в расшифровке.

Расшифровка не говорит, где человек находится и что делает: «пошли» может
означать и выход из машины, и вход в танк. Монтажёр это видит. Даём кадр
раз в 12 секунд видео-языковой модели и просим одну строку описания.

Раньше признаки такой модели в бустинге не сработали — они на 85 %
объяснялись тегами CLIP. Здесь другое применение: описание идёт не числом
в модель, а текстом в контекст языковой модели, которая принимает решение.

Запускать на GPU-сервере:
  python3 vlm_describe.py --slot 5
"""
import json, os, sys, time, argparse, subprocess
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FRAMES = os.path.join(os.environ.get("SEM_WORK", "work/window"), "frames_desc")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
TMAX = 78
PROMPT = ("Опиши кадр уличного стрима одной строкой на русском: где это "
          "(улица, машина, магазин, кафе, помещение, природа), сколько людей "
          "в кадре, что они делают. Без вступлений, до 15 слов.")


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    d = f"{FRAMES}/s{a.slot}"
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    out_path = f"{OUT}/desc_s{a.slot}.json"
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path))
        print(f"уже есть {len(done)} описаний", flush=True)
    todo = [f for f in files if f not in done]
    if not todo:
        print("всё готово"); return

    proc = AutoProcessor.from_pretrained(MODEL)
    mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    print(f"слот {a.slot}: {len(todo)} кадров", flush=True)

    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        msgs = []
        for f in chunk:
            msgs.append([{"role": "user", "content": [
                {"type": "image", "image": Image.open(f"{d}/{f}").convert("RGB")},
                {"type": "text", "text": PROMPT}]}])
        texts = [proc.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=True) for m in msgs]
        imgs = [[Image.open(f"{d}/{f}").convert("RGB")] for f in chunk]
        inp = proc(text=texts, images=imgs, return_tensors="pt",
                   padding=True).to(mdl.device)
        with torch.no_grad():
            g = mdl.generate(**inp, max_new_tokens=48, do_sample=False)
        for j, f in enumerate(chunk):
            n = inp["input_ids"].shape[1]
            done[f] = proc.decode(g[j][n:], skip_special_tokens=True).strip()
        if i % (a.batch * 10) == 0:
            el = time.time() - t0
            t = gpu_temp()
            print(f"  {i+len(chunk)}/{len(todo)}  {t} °C  "
                  f"{el/max(i+len(chunk),1):.2f} с/кадр", flush=True)
            json.dump(done, open(out_path, "w"), ensure_ascii=False)
            while t > TMAX:
                print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                time.sleep(60); t = gpu_temp()
    json.dump(done, open(out_path, "w"), ensure_ascii=False)
    print(f"слот {a.slot}: готово за {time.time()-t0:.0f} с -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
