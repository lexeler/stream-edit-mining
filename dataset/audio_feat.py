#!/usr/bin/env python3
"""
Звуковые признаки, которых в конвейере не было.

  CLAP  — это CLIP для звука: зеро-шот ответы на вопросы к куску звука.
          Смех, толпа, музыка, двигатель, тишина, крик. Прямые маркеры
          «живого» и «мёртвого» эфира, которых нет ни в тексте, ни в громкости.
  x-vec — отпечаток голоса каждые 2 секунды (WavLM). Из него считаем
          ИЗМЕНЧИВОСТЬ голоса в окне: если голос всё время один — человек
          говорит сам с собой; если меняется — идёт разговор с людьми.
          Для IRL-стрима это, похоже, главный макро-сигнал.

Ни torchaudio, ни librosa не нужны: FLAC читаем через ffmpeg в сырой PCM.
"""
import os, sys, time, subprocess
import numpy as np
import torch

AUD = os.path.join(os.environ.get("SEM_WORK", "work/window"), "aud")
OUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SR = 16000
CLAP_WIN, CLAP_HOP = 10.0, 5.0      # окна CLAP
XV_WIN, XV_HOP = 2.0, 2.0           # окна отпечатка голоса
TMAX = 80

TAGS = {
    "смех":     "people laughing out loud",
    "толпа":    "many people talking, crowd noise",
    "разговор": "two people having a conversation",
    "один":     "a single person speaking alone",
    "крик":     "someone shouting or screaming",
    "музыка":   "music playing",
    "машина":   "a car engine running while driving",
    "улица":    "outdoor street noise, traffic",
    "ветер":    "wind blowing into the microphone",
    "тишина":   "silence, almost no sound",
    "шаги":     "footsteps walking",
    "еда":      "eating, dishes, restaurant sounds",
    "аплодис":  "applause and cheering",
    "телефон":  "sound from a phone or TV speaker",
    "вода":     "water, river or rain sounds",
    "стук":     "knocking, banging, impact noise",
}


def read_flac(path, sr=SR):
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path,
                        "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                       capture_output=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def gpu_temp():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True)
    try:
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def cool():
    t = gpu_temp()
    while t > TMAX:
        print(f"  ГОРЯЧО {t} °C — пауза 60 с", flush=True)
        time.sleep(60); t = gpu_temp()


def run_clap(x, dev, sr=48000):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    pr = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    names = list(TAGS)
    with torch.no_grad():
        t = pr(text=[TAGS[k] for k in names], return_tensors="pt", padding=True)
        T = m.get_text_features(**{k: v.to(dev) for k, v in t.items()})
        T = torch.nn.functional.normalize(T, dim=-1)
    w, h = int(CLAP_WIN * sr), int(CLAP_HOP * sr)
    n = max(1, (len(x) - w) // h + 1)
    out = []
    B = 16
    for i in range(0, n, B):
        chunks = [x[j * h:j * h + w] for j in range(i, min(i + B, n))]
        chunks = [np.pad(c, (0, w - len(c))) if len(c) < w else c for c in chunks]
        with torch.no_grad():
            f = pr(audios=chunks, sampling_rate=sr, return_tensors="pt")
            e = m.get_audio_features(**{k: v.to(dev) for k, v in f.items()})
            e = torch.nn.functional.normalize(e, dim=-1)
            out.append((e @ T.T).cpu().numpy().astype(np.float16))
        if (i // B) % 100 == 0:
            print(f"    CLAP {i}/{n}  {gpu_temp()} °C", flush=True); cool()
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out), names


def run_xvec(x, dev):
    from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
    name = "microsoft/wavlm-base-plus-sv"
    m = WavLMForXVector.from_pretrained(name).to(dev).eval().half()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    w, h = int(XV_WIN * SR), int(XV_HOP * SR)
    n = max(1, (len(x) - w) // h + 1)
    out = []
    B = 64
    for i in range(0, n, B):
        chunks = [x[j * h:j * h + w] for j in range(i, min(i + B, n))]
        chunks = [np.pad(c, (0, w - len(c))) if len(c) < w else c for c in chunks]
        with torch.no_grad():
            f = fe(chunks, sampling_rate=SR, return_tensors="pt", padding=True)
            e = m(input_values=f["input_values"].to(dev).half()).embeddings
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out.append(e.cpu().numpy().astype(np.float16))
        if (i // B) % 200 == 0:
            print(f"    xvec {i}/{n}  {gpu_temp()} °C", flush=True); cool()
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    part = "both"
    args = sys.argv[1:]
    if args and args[0] in ("clap", "xvec", "both"):
        part, args = args[0], args[1:]
    slots = args or ["1", "2", "3", "4", "5", "6", "7"]
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"считаем на {dev}", flush=True)
    for s in slots:
        t0 = time.time()
        if part in ("xvec", "both"):
            p = f"{OUT}/xvec_s{s}.npz"
            f = f"{AUD}/s{s}.flac"
            if os.path.exists(p):
                print(f"слот {s}: xvec уже есть", flush=True)
            elif not os.path.exists(f):
                print(f"слот {s}: звука 16к нет", flush=True)
            else:
                x = read_flac(f, SR)
                print(f"слот {s}: {len(x)/SR/60:.1f} мин, xvec", flush=True)
                xv = run_xvec(x, dev)
                d1 = 1 - (xv[1:].astype(np.float32) * xv[:-1].astype(np.float32)).sum(1)
                d1 = np.concatenate([[0], d1]).astype(np.float16)
                np.savez(p, xv=xv, dxv=d1, xv_hop=XV_HOP)
                print(f"слот {s}: xvec за {time.time()-t0:.0f} с", flush=True)
        if part in ("clap", "both"):
            p = f"{OUT}/clap_s{s}.npz"
            f = f"{AUD}/s{s}.48.flac"
            if os.path.exists(p):
                print(f"слот {s}: clap уже есть", flush=True)
            elif not os.path.exists(f):
                print(f"слот {s}: звука 48к нет", flush=True)
            else:
                x = read_flac(f, 48000)
                print(f"слот {s}: {len(x)/48000/60:.1f} мин, clap", flush=True)
                clap, names = run_clap(x, dev, 48000)
                np.savez(p, clap=clap, names=names, clap_hop=CLAP_HOP)
                print(f"слот {s}: clap за {time.time()-t0:.0f} с", flush=True)
    print("готово")


if __name__ == "__main__":
    main()
