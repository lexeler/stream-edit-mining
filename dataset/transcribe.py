#!/usr/bin/env python3
"""
Полная расшифровка видео в ленту событий «слово — пауза».

Для каждого файла: звук -> FLAC 16 кГц моно -> ASR-сервер -> GigaAM v3
-> разбор таймкодов -> JSON.
"""
import json, os, re, subprocess, sys, time

OUT = os.environ.get("SEM_DATASET", "work/dataset")
TMP = os.environ.get("SEM_TMP", "work/tmp")
SOURCES = os.environ.get("SEM_SOURCES", "work/sources")   # исходные видео
ASR_HOST = os.environ.get("SEM_ASR_HOST", "asr")         # сервер с GigaAM
ASR_DIR = os.environ.get("SEM_ASR_DIR", "~/speech")

JOBS = [
    ("stream_2026-06-30_varenikovskiy",
     os.path.join(os.environ.get("SEM_STREAMS", "work/streams"), "7/streams/full_video.mp4"),
     "полный стрим 30 июня (слот 7) — исходник ролика"),
    ("stream_2026-07-01_ded",
     f"{SOURCES}/stream_2026-07-01.mp4",
     "полный стрим 1 июля"),
    ("edit_youtube_futbolnaya-zaruba",
     f"{SOURCES}/edit_youtube_futbolnaya-zaruba.mp4",
     "смонтированный выпуск с YouTube"),
]

MIN_SIL = 0.04      # паузы короче не пишем (сетка модели 40 мс)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", path]).stdout.strip())


def build(name, video, note):
    print(f"\n{'='*64}\n{name}\n{'='*64}", flush=True)
    dur = duration(video)
    print(f"  длительность {dur/3600:.2f} ч", flush=True)

    flac = f"{TMP}/{name}.flac"
    if not os.path.exists(flac):
        t0 = time.time()
        r = sh(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1",
                "-ar", "16000", "-c:a", "flac", "-compression_level", "5",
                "-y", flac])
        if r.returncode:
            print("  ОШИБКА ffmpeg:", r.stderr[:300]); return None
        print(f"  звук: {os.path.getsize(flac)/2**20:.0f} МБ за {time.time()-t0:.0f} с", flush=True)

    t0 = time.time()
    if sh(["scp", "-q", flac, f"{ASR_HOST}:/tmp/{name}.flac"]).returncode:
        print("  ОШИБКА передачи"); return None
    print(f"  передано за {time.time()-t0:.0f} с", flush=True)

    t0 = time.time()
    r = sh(["ssh", "-o", "BatchMode=yes", ASR_HOST,
            f"cd {ASR_DIR} && ./.venv/bin/python transcribe.py "
            f"/tmp/{name}.flac --timestamps 2>&1"])
    sh(["ssh", "-o", "BatchMode=yes", ASR_HOST, f"rm -f /tmp/{name}.flac"])
    print(f"  распознано за {time.time()-t0:.0f} с", flush=True)

    words = []
    for ln in r.stdout.splitlines():
        m = re.match(r"\s*\[\s*([\d.]+)\s*-\s*([\d.]+)\]\s+(.*\S)\s*$", ln)
        if m:
            words.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    if not words:
        print("  НЕ РАСПОЗНАЛОСЬ:", r.stdout[-400:]); return None

    items, prev = [], 0.0
    for s, e, w in words:
        if s - prev >= MIN_SIL:
            items.append({"type": "silence", "start": round(prev, 2), "end": round(s, 2)})
        items.append({"type": "word", "word": w, "start": round(s, 2), "end": round(e, 2)})
        prev = max(prev, e)
    if dur - prev >= MIN_SIL:
        items.append({"type": "silence", "start": round(prev, 2), "end": round(dur, 2)})

    nw = sum(1 for i in items if i["type"] == "word")
    ns = len(items) - nw
    speech = sum(i["end"] - i["start"] for i in items if i["type"] == "word")
    doc = {
        "source": os.path.basename(video),
        "note": note,
        "duration_sec": round(dur, 3),
        "model": "GigaAM v3 e2e_rnnt",
        "time_grid_sec": 0.04,
        "words_total": nw,
        "silences_total": ns,
        "speech_sec": round(speech, 2),
        "items": items,
    }
    p = f"{OUT}/{name}.json"
    json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1)
    with open(f"{OUT}/{name}.txt", "w") as f:
        f.write(" ".join(i["word"] for i in items if i["type"] == "word") + "\n")
    print(f"  слов {nw}, пауз {ns}, речи {speech/60:.1f} мин из {dur/60:.1f}", flush=True)
    print(f"  -> {os.path.basename(p)} ({os.path.getsize(p)/2**20:.1f} МБ)", flush=True)
    os.remove(flac)
    return doc


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    done = []
    for name, video, note in JOBS:
        if not os.path.exists(video):
            print(f"НЕТ ФАЙЛА: {video}"); continue
        d = build(name, video, note)
        if d:
            done.append((name, d))
    print(f"\n{'='*64}\nГОТОВО: {len(done)} файлов")
    for n, d in done:
        print(f"  {n}: {d['words_total']} слов, {d['silences_total']} пауз, "
              f"речь {d['speech_sec']/d['duration_sec']*100:.0f}% времени")
