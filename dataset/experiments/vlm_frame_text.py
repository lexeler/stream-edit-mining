#!/usr/bin/env python3
"""
Кадр и расшифровка ВМЕСТЕ — то, что видит и слышит человек за монтажным столом.

До сих пор картинку и текст модели получали порознь: CLIP и Qwen3-VL видели
кадр без слов, текстовый судья читал слова без кадра. Между тем половина
смысла в связке: «пустая улица» плюс «смотрите, что нашёл» — это событие,
а «пустая улица» плюс молчание — под нож.

  vlm_frame_text.py [шаг_с] [слоты...]
"""
import os, sys, time, glob, json, collections, subprocess
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

VID = os.path.join(os.environ.get("SEM_WORK", "work/window"), "vid")
DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TMAX = 80
FPS = 2.0
CTX = 30.0

SYSTEM = (
    "Ты помогаешь монтажёру ютуб-канала. Это уличный стрим на три часа, "
    "из него делают выпуск на час: берут живое, смешное, события "
    "и разговоры с людьми, выбрасывают дорогу, молчание и пустой трёп. "
    "Тебе дают кадр и то, что говорят в этот момент. "
    "Отвечай одним словом: Да или Нет."
)

QUESTIONS = [
    ("keep",   "Этот момент стоит оставить в выпуске?"),
    ("event",  "Здесь происходит событие или действие?"),
    ("people", "Он общается с другими людьми?"),
    ("dead",   "Это пустой проходной момент?"),
    ("funny",  "Здесь смешно?"),
]


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def load_words(slot):
    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
        w = [(r["start"], r["word"]) for r in rows
             if r["slot"] == slot and r["type"] == "word"]
        if w:
            return np.array([x[0] for x in w]), [x[1] for x in w]
    return np.zeros(1), []


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

    for s in slots:
        p = f"{DATA}/vlm3_s{s}.npz"
        if os.path.exists(p):
            print(f"слот {s}: уже есть", flush=True); continue
        files = sorted(glob.glob(f"{VID}/s{s}/f*.jpg"))[::step]
        if not files:
            print(f"слот {s}: кадров нет", flush=True); continue
        ws, words = load_words(s)
        out = np.zeros((len(files), len(QUESTIONS)), np.float32)
        t0 = time.time()
        for i, f in enumerate(files):
            t = i * hop
            lo = int(np.searchsorted(ws, t - CTX))
            hi = int(np.searchsorted(ws, t + CTX))
            said = " ".join(words[lo:hi])[:700] or "(молчание)"
            img = Image.open(f).convert("RGB")
            txts = []
            for _, q in QUESTIONS:
                msg = [{"role": "system",
                        "content": [{"type": "text", "text": SYSTEM}]},
                       {"role": "user",
                        "content": [{"type": "image"},
                                    {"type": "text",
                                     "text": f"Говорят: «{said}»\n\n{q}"}]}]
                txts.append(proc.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True))
            inp = proc(text=txts, images=[img] * len(txts),
                       return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                lg = mdl(**inp).logits[:, -1].float()
            out[i] = (torch.logsumexp(lg[:, YES], 1) -
                      torch.logsumexp(lg[:, NO], 1)).cpu().numpy()
            if i % 100 == 0:
                tt = gpu_temp()
                print(f"  слот {s}: {i}/{len(files)}  {tt} °C  "
                      f"{(time.time()-t0)/max(i,1):.2f} с/кадр", flush=True)
                while tt > TMAX:
                    print(f"  ГОРЯЧО {tt} °C — пауза 60 с", flush=True)
                    time.sleep(60); tt = gpu_temp()
        np.savez(p, scores=out, names=[k for k, _ in QUESTIONS], hop=hop)
        print(f"слот {s}: {len(files)} кадров за {time.time()-t0:.0f} с",
              flush=True)
    print("готово")


if __name__ == "__main__":
    main()
