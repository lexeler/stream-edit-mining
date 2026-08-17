#!/usr/bin/env python3
"""
Видео-признаки через CLIP. Считается на GPU-сервере, кадры уже извлечены (2 к/с).

Три группы на выходе:
  emb    — вектор кадра (768), нужен для сжатия и для похожести во времени;
  tags   — зеро-шот ответы на 24 понятных вопроса к кадру. При 2400 решениях
           монтажёра осмысленные признаки надёжнее сжатого шума;
  stasis — визуальный застой: похожесть кадра на то, что было 30 с, 2 мин
           и 5 мин назад. Если картинка не меняется — вероятен скучный кусок.

Кадр k (f000001 = k=1) соответствует времени (k-1)/2 секунды.
"""
import os, sys, time, glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VID = os.path.join(os.environ.get("SEM_WORK", "work/window"), "vid")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "openai/clip-vit-large-patch14"
BATCH = 96
TMAX = 80          # выше этой температуры делаем паузу

PROMPTS = [
    "a group of people talking together",
    "a crowd of many people",
    "one person alone",
    "people laughing and having fun",
    "a close-up of a human face",
    "a man talking to the camera, selfie style",
    "the inside of a car while driving",
    "an empty road or highway",
    "a city street with buildings",
    "indoors, inside a room or apartment",
    "outdoors in nature, trees and grass",
    "an abandoned or ruined building",
    "a shop, store or supermarket",
    "a cafe, bar or restaurant table with food",
    "night time, dark scene",
    "bright daylight outdoors",
    "a phone or computer screen",
    "a blurry shaky out-of-focus image",
    "the ground or the sky, camera pointing nowhere",
    "water, a river, a lake or the sea",
    "a dog, cat or other animal",
    "people playing sport or doing physical activity",
    "someone sitting still doing nothing",
    "a vehicle, bicycle or motorcycle",
]


class Frames(torch.utils.data.Dataset):
    def __init__(self, files, proc):
        self.files = files
        self.mean = torch.tensor(proc.image_processor.image_mean).view(3, 1, 1)
        self.std = torch.tensor(proc.image_processor.image_std).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB").resize((224, 224),
                                                             Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std


def gpu_temp():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True)
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def main():
    slots = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    dev = "cuda:0"
    model = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    with torch.no_grad():
        tk = proc(text=PROMPTS, return_tensors="pt", padding=True)
        T = model.get_text_features(**{k: v.to(dev) for k, v in tk.items()})
        T = (T / T.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    for s in slots:
        out = f"{OUT}/clip_s{s}.npz"
        if os.path.exists(out):
            print(f"слот {s}: уже есть", flush=True); continue
        files = sorted(glob.glob(f"{VID}/s{s}/f*.jpg"))
        if not files:
            print(f"слот {s}: кадров нет", flush=True); continue
        dl = torch.utils.data.DataLoader(Frames(files, proc), batch_size=BATCH,
                                         num_workers=4, pin_memory=True)
        t0 = time.time(); embs = []
        for bi, x in enumerate(dl):
            with torch.no_grad():
                e = model.get_image_features(pixel_values=x.to(dev, torch.float16))
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.float().cpu().numpy().astype(np.float16))
            if bi % 200 == 0:
                t = gpu_temp()
                print(f"  слот {s}: {bi*BATCH}/{len(files)}  {t} °C", flush=True)
                while t > TMAX:
                    print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
                    time.sleep(60); t = gpu_temp()
        E = np.concatenate(embs)
        tags = (E.astype(np.float32) @ T.T).astype(np.float16)

        # визуальный застой: похожесть на среднее по прошлому окну
        F = E.astype(np.float32)
        st = {}
        for name, sec in (("s30", 30), ("s120", 120), ("s300", 300)):
            k = int(sec * 2)
            c = np.cumsum(np.vstack([np.zeros((1, F.shape[1]), np.float32), F]), 0)
            lo = np.clip(np.arange(len(F)) - k, 0, None)
            hi = np.arange(len(F))
            n = np.maximum(hi - lo, 1)
            past = (c[hi] - c[lo]) / n[:, None]
            past /= (np.linalg.norm(past, axis=1, keepdims=True) + 1e-6)
            st[name] = (F * past).sum(1).astype(np.float16)
        np.savez(out, emb=E, tags=tags, **st, n=len(F))
        print(f"слот {s}: {len(F)} кадров за {time.time()-t0:.0f} с -> {out}",
              flush=True)
    print("готово")


if __name__ == "__main__":
    main()
