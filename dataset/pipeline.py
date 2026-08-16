#!/usr/bin/env python3
"""
Полный конвейер: стрим -> обучающая выборка.

Ключевой принцип: элементы строятся ТОЛЬКО из звука, без оглядки на монтажный
лист. Иначе границы элементов сами выдают ответ — на тесте это недопустимо,
на обучении это утечка.

Метки навешиваются сверху: пересёкся с монтажом -> used=true.
"""
import json, math, os, re, subprocess, sys, time
import numpy as np

BASE = os.environ.get("SEM_STREAMS", "work/streams")
OUT = os.environ.get("SEM_DATASET", "work/dataset")
TMP = os.path.join(os.environ.get("SEM_TMP", "work/tmp"), "pipe")
ASR_HOST = os.environ.get("SEM_ASR_HOST", "asr")         # сервер с GigaAM
ASR_DIR = os.environ.get("SEM_ASR_DIR", "~/speech")

SLOTS = {
    "1": ("slot1_2026-07-30_futbol-mozgi", "train"),
    "2": ("slot2_2026-07-31_gulyanka", "test"),
    "3": ("slot3_2026-08-03_tarzanka-zabroshka", "train"),
    "4": ("slot4_2026-08-04_razvlekalovka", "train"),
    "5": ("slot5_2026-06-12_matiz", "train"),
    "6": ("slot6_2026-06-30_2mp4", "train"),
    "7": ("slot7_2026-06-30_varenikovskiy", "train"),
}

SR8, WIN, HOP = 8000, 256, 80
LONG_W, MIN_GAP, REL, ABS_ = 5.0, 0.30, 0.08, 0.005
CHUNK, MIN_TAIL = 5.0, 1.0
DB_LO, DB_HI = -60.0, 0.0
FILLER = {"ну", "вот", "это", "как", "бы", "типа", "короче", "э", "а", "блин",
          "значит", "самое", "такой", "там", "ээ", "мм", "эм"}
MAT = re.compile(r"(бля|хуй|хуе|хуё|пизд|ебан|ёбан|еба|ёба|сук[аи]|нахуй|пидор|говн|муд)", re.I)


def sh(c, **k):
    return subprocess.run(c, capture_output=True, text=True, **k)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. звук
def extract(video, name):
    flac, raw = f"{TMP}/{name}.flac", f"{TMP}/{name}.raw"
    if not os.path.exists(flac):
        sh(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "flac", "-compression_level", "5", "-y", flac])
    if not os.path.exists(raw):
        sh(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR8),
            "-f", "s16le", "-y", raw])
    return flac, raw


# ------------------------------------------------------- 2. распознавание
def recognise(flac, name):
    cache = f"{TMP}/{name}.words.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    sh(["scp", "-q", flac, f"{ASR_HOST}:/tmp/{name}.flac"])
    r = sh(["ssh", "-o", "BatchMode=yes", ASR_HOST,
            f"cd {ASR_DIR} && ./.venv/bin/python transcribe.py "
            f"/tmp/{name}.flac --timestamps 2>&1"])
    sh(["ssh", "-o", "BatchMode=yes", ASR_HOST, f"rm -f /tmp/{name}.flac"])
    w = []
    for ln in r.stdout.splitlines():
        m = re.match(r"\s*\[\s*([\d.]+)\s*-\s*([\d.]+)\]\s+(.*\S)\s*$", ln)
        if m:
            w.append([float(m.group(1)), float(m.group(2)), m.group(3)])
    if not w:
        log("   НЕ РАСПОЗНАЛОСЬ:", r.stdout[-300:])
        return None
    json.dump(w, open(cache, "w"), ensure_ascii=False)
    return w


# ------------------------------------------------- 3. лента и её чистка
def silent_runs(seg):
    w = int(0.025 * SR8)
    n = len(seg) // w
    if n < 2:
        return []
    rms = np.sqrt((seg[:n * w].reshape(n, w) ** 2).mean(axis=1))
    thr = max(rms.max() * REL, ABS_)
    quiet = rms < thr
    runs, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.025 >= MIN_GAP:
                runs.append((i * 0.025, j * 0.025))
            i = j
        else:
            i += 1
    return runs


def build_items(words, dur, x):
    items, prev = [], 0.0
    for s, e, t in words:
        if s - prev >= 0.04:
            items.append({"type": "silence", "start": round(prev, 2), "end": round(s, 2)})
        items.append({"type": "word", "word": t, "start": round(s, 2), "end": round(e, 2)})
        prev = max(prev, e)
    if dur - prev >= 0.04:
        items.append({"type": "silence", "start": round(prev, 2), "end": round(dur, 2)})

    # длинные слова: тишина внутри -> отдельный блок
    out = []
    for it in items:
        d = it["end"] - it["start"]
        if it["type"] != "word" or d <= LONG_W:
            out.append(it)
            continue
        runs = silent_runs(x[int(it["start"] * SR8):int(it["end"] * SR8)])
        if not runs:
            out.append(it)
            continue
        first = runs[0][0] if runs[0][0] >= 0.08 else max(0.08, runs[0][1])
        we = round(it["start"] + first, 2)
        out.append({"type": "word", "word": it["word"], "start": it["start"], "end": we})
        out.append({"type": "silence", "start": we, "end": it["end"]})

    # слить соседние тишины
    m = []
    for it in out:
        if (m and it["type"] == "silence" and m[-1]["type"] == "silence"
                and abs(m[-1]["end"] - it["start"]) < 1e-6):
            m[-1]["end"] = it["end"]
        else:
            m.append(dict(it))

    # длинные тишины -> куски по 5 с
    fin = []
    for it in m:
        d = it["end"] - it["start"]
        if it["type"] != "silence" or d <= CHUNK:
            fin.append(it)
            continue
        xs, t = [], it["start"]
        while it["end"] - t > CHUNK:
            xs.append((t, t + CHUNK))
            t += CHUNK
        if it["end"] - t < MIN_TAIL and xs:
            xs[-1] = (xs[-1][0], it["end"])
        else:
            xs.append((t, it["end"]))
        for a, b in xs:
            fin.append({"type": "silence", "start": round(a, 2), "end": round(b, 2)})
    return fin


# ------------------------------------------------------------ 4. признаки
def frame_feats(x):
    n = 1 + (len(x) - WIN) // HOP
    w = np.hanning(WIN).astype(np.float32)
    fq = np.fft.rfftfreq(WIN, 1 / SR8)
    low = (fq >= 80) & (fq <= 500)
    o = {k: np.zeros(n, dtype=np.float32)
         for k in ("rms", "zcr", "centroid", "flatness", "lowfreq")}
    for s in range(0, n, 100_000):
        c = min(100_000, n - s)
        idx = (np.arange(c) * HOP)[:, None] + np.arange(WIN)[None, :] + s * HOP
        fr = x[idx]
        o["rms"][s:s + c] = np.sqrt((fr ** 2).mean(axis=1))
        o["zcr"][s:s + c] = np.diff(np.signbit(fr), axis=1).sum(axis=1) / WIN
        mag = np.abs(np.fft.rfft(fr * w, axis=1)) + 1e-10
        tot = mag.sum(axis=1)
        o["centroid"][s:s + c] = (mag * fq).sum(axis=1) / tot / (SR8 / 2)
        o["flatness"][s:s + c] = np.exp(np.log(mag).mean(axis=1)) / (tot / mag.shape[1])
        o["lowfreq"][s:s + c] = mag[:, low].sum(axis=1) / tot
    return o, n


def add_features(items, x, dur):
    ff, nf = frame_feats(x)
    cum = {k: np.concatenate(([0.0], np.cumsum(v.astype(np.float64))))
           for k, v in ff.items()}
    cum["rms2"] = np.concatenate(([0.0], np.cumsum(ff["rms"].astype(np.float64) ** 2)))

    def agg(a, b):
        i, j = int(a * 100), min(int(b * 100), nf)
        if j <= i:
            i, j = min(i, nf - 1), min(i + 1, nf)
        m = j - i
        r = (cum["rms"][j] - cum["rms"][i]) / m
        r2 = (cum["rms2"][j] - cum["rms2"][i]) / m
        return (r, math.sqrt(max(r2 - r * r, 0.0)),
                (cum["zcr"][j] - cum["zcr"][i]) / m,
                (cum["centroid"][j] - cum["centroid"][i]) / m,
                (cum["flatness"][j] - cum["flatness"][i]) / m,
                (cum["lowfreq"][j] - cum["lowfreq"][i]) / m)

    N = len(items)
    starts = np.array([i["start"] for i in items])
    dr = np.array([i["end"] - i["start"] for i in items])
    isw = np.array([1 if i["type"] == "word" else 0 for i in items])
    cs_sp = np.concatenate(([0.0], np.cumsum(np.where(isw == 1, dr, 0.0))))
    cs_du = np.concatenate(([0.0], np.cumsum(dr)))
    rows = [agg(i["start"], i["end"]) for i in items]
    vol = np.array([min(1.0, max(0.0, (20 * math.log10(max(r[0], 1e-6)) - DB_LO)
                                 / (DB_HI - DB_LO))) for r in rows])
    cs_v = np.concatenate(([0.0], np.cumsum(vol)))
    cs_p = np.concatenate(([0.0], np.cumsum(1 - isw)))

    dens, vrel = {}, {}
    for s in (5, 15, 60):
        lo = np.searchsorted(starts, starts - s)
        hi = np.searchsorted(starts, starts + s)
        dens[s] = (cs_sp[hi] - cs_sp[lo]) / np.maximum(cs_du[hi] - cs_du[lo], 1e-6)
        vrel[s] = vol - (cs_v[hi] - cs_v[lo]) / np.maximum(hi - lo, 1)
        if s == 15:
            pdens = (cs_p[hi] - cs_p[lo]) / 30.0

    blk = np.zeros(N)
    cur = items[0]["start"]
    for k in range(N):
        blk[k] = cur          # сначала записать, потом двигать отсчёт:
        if items[k]["type"] == "silence" and dr[k] > 1.0:
            cur = items[k]["end"]   # иначе сама пауза получит отрицательное время

    for k, it in enumerate(items):
        w = it.get("word", "")
        wl = w.strip(".,!?»«:;()-—").lower()
        p2 = items[k - 2].get("word", "") if k > 1 else ""
        r = rows[k]
        it["f"] = {
            "dur": round(float(dr[k]), 3), "is_word": int(isw[k]),
            "rel_pos": round(float((it["start"] + it["end"]) / 2 / dur), 4),
            "vol": round(float(vol[k]), 3), "vol_std": round(float(r[1]), 4),
            "zcr": round(float(r[2]), 4), "centroid": round(float(r[3]), 4),
            "flatness": round(float(r[4]), 4), "voice": round(float(r[5]), 4),
            "pause_before": round(float(dr[k - 1]) if k and isw[k - 1] == 0 else 0.0, 3),
            "pause_after": round(float(dr[k + 1]) if k + 1 < N and isw[k + 1] == 0 else 0.0, 3),
            "dens5": round(float(dens[5][k]), 3), "dens15": round(float(dens[15][k]), 3),
            "dens60": round(float(dens[60][k]), 3),
            "vrel5": round(float(vrel[5][k]), 3), "vrel15": round(float(vrel[15][k]), 3),
            "vrel60": round(float(vrel[60][k]), 3),
            "pause_dens": round(float(pdens[k]), 3),
            "block_pos": round(float(it["start"] - blk[k]), 2),
            "d_dens": round(float(dens[5][min(k + 1, N - 1)] - dens[5][max(k - 1, 0)]), 3),
            "d_vol": round(float(vol[min(k + 1, N - 1)] - vol[max(k - 1, 0)]), 3),
            "wlen": len(wl), "is_filler": int(wl in FILLER),
            "has_mat": int(bool(MAT.search(wl))),
            "ends_sent": int(w.endswith((".", "!", "?", "…"))),
            "is_question": int(w.endswith("?")),
            "is_caps": int(bool(w[:1].isupper())),
            "is_repeat": int(bool(wl) and wl == p2.strip(".,!?»«:;()-—").lower()),
        }
    return items


# -------------------------------------------------------------- 5. метки
def label(items, edl_path):
    segs = sorted((s["start"], s["end"])
                  for s in json.load(open(edl_path))["segments"])
    import bisect
    st = [a for a, _ in segs]
    out = []
    for it in items:
        a, b = it["start"], it["end"]
        i = max(0, bisect.bisect_right(st, a) - 1)
        ov = 0.0
        while i < len(segs) and segs[i][0] < b:
            ov += max(0.0, min(b, segs[i][1]) - max(a, segs[i][0]))
            i += 1
        d = b - a
        out.append((ov > 0, round(ov / d, 3) if d > 0 else 0.0))
    return out, len(segs), sum(b - a for a, b in segs)


# ---------------------------------------------------------------- главная
def process(slot, name, split):
    video = f"{BASE}/{slot}/streams/full_video.mp4"
    edl = f"{BASE}/{slot}/narezki/timecodes.json"
    log(f"\n{'='*66}\nСЛОТ {slot} -> {name}  [{split}]\n{'='*66}")
    dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", video]).stdout.strip())
    log(f"  длительность {dur/3600:.2f} ч")

    t0 = time.time()
    flac, raw = extract(video, name)
    log(f"  звук извлечён за {time.time()-t0:.0f} с")

    t0 = time.time()
    words = recognise(flac, name)
    if not words:
        return None
    log(f"  распознано {len(words)} слов за {time.time()-t0:.0f} с")

    x = np.fromfile(raw, dtype="<i2").astype(np.float32) / 32768
    items = build_items(words, dur, x)
    nw = sum(1 for i in items if i["type"] == "word")
    log(f"  лента: {len(items)} элементов ({nw} слов, {len(items)-nw} тишины)")

    t0 = time.time()
    items = add_features(items, x, dur)
    log(f"  признаки посчитаны за {time.time()-t0:.0f} с")
    del x

    lab, nseg, edur = label(items, edl)
    used = sum(1 for u, _ in lab if u)
    tused = sum(i["end"] - i["start"] for i, (u, _) in zip(items, lab) if u)
    log(f"  монтаж: {nseg} отрезков, {edur/60:.1f} мин")
    log(f"  метки: used {used}/{len(items)} = {used/len(items)*100:.1f}%, "
        f"время {tused/60:.1f} мин (переоценка {(tused-edur)/60:+.1f})")

    head = {"slot": slot, "split": split, "source": os.path.basename(video),
            "duration_sec": round(dur, 3), "model": "GigaAM v3 e2e_rnnt",
            "items_total": len(items), "words_total": nw,
            "silences_total": len(items) - nw,
            "pipeline": ("слова >5с разрезаны по внутренней тишине; тишина >5с "
                         "нарезана по 5с; элементы построены только из звука"),
            "features": list(items[0]["f"].keys()),
            "labels_note": ("used = задето хотя бы частично; "
                            "used_half = вошло больше половины (точнее совпадает "
                            "с монтажом, но правило строже)")}

    json.dump({**head, "items": [{k: v for k, v in i.items()} for i in items]},
              open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    if split == "train":
        li = [{**i, "used": u, "used_frac": f, "used_half": bool(f >= 0.5)}
              for i, (u, f) in zip(items, lab)]
        json.dump({**head, "edl_segments": nseg, "edl_duration_sec": round(edur, 2),
                   "items": li}, open(f"{OUT}/{name}.labeled.json", "w"),
                  ensure_ascii=False)
    else:
        json.dump({**head, "edl_segments": nseg, "edl_duration_sec": round(edur, 2),
                   "answers": [{"i": k, "start": i["start"], "end": i["end"],
                                "used": u, "used_frac": f,
                                "used_half": bool(f >= 0.5)}
                               for k, (i, (u, f)) in enumerate(zip(items, lab))]},
                  open(f"{OUT}/{name}.answers.json", "w"), ensure_ascii=False)
    for p in (flac, raw):
        if os.path.exists(p):
            os.remove(p)
    return {"slot": slot, "name": name, "split": split, "items": len(items),
            "used": used, "dur": dur}


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(SLOTS)
    done = []
    for s in only:
        n, sp = SLOTS[s]
        if os.path.exists(f"{OUT}/{n}.json") and os.path.exists(
                f"{OUT}/{n}.{'labeled' if sp=='train' else 'answers'}.json"):
            log(f"слот {s}: уже готов, пропускаю")
            continue
        r = process(s, n, sp)
        if r:
            done.append(r)
    log(f"\nобработано: {len(done)}")
