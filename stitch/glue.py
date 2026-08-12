#!/usr/bin/env python3
"""
Склейка кусков одного стрима в единый файл — без перекодирования.

    python3 glue.py <папка> [-o full_video.mp4] [--dry-run]

Что делает сам, без участия человека:
  1. определяет порядок кусков по меткам времени в именах;
  2. находит перехлёсты между соседями по звуку (взаимная корреляция);
  3. сажает точки реза на ключевые кадры — это и позволяет обойтись
     стрим-копированием вместо перекодирования;
  4. склеивает через ffmpeg concat, видео и звук копируются байт в байт;
  5. ИЗМЕРЯЕТ фактическое положение каждого куска в результате
     и пишет offsets.json — он нужен для пересчёта таймкодов.

Почему именно стрим-копирование: см. README, раздел «Тупик с перекодированием звука».
"""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np

SR = 16000          # частота дискретизации для анализа звука
WIN = 400.0         # окно поиска перехлёста, с
MIN_OVERLAP = 0.20  # меньше этого перехлёстом не считаем, с
NCC_MIN = 0.75      # порог первичной корреляции
CONFIRM_MIN = 0.80  # порог подтверждения на контрольных окнах
CONFIRM_PROBES = (0.15, 0.35, 0.55, 0.75, 0.92)   # где брать контрольные окна


# ---------------------------------------------------------------- утилиты

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    j = json.loads(_run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                         "-of", "json", path]).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    num, den = v["r_frame_rate"].split("/")
    return {
        "dur": float(j["format"]["duration"]),
        "fps": int(num) / int(den),
        "nframes": int(v.get("nb_frames") or 0),
        "w": v["width"], "h": v["height"],
        "vcodec": v["codec_name"], "pix": v.get("pix_fmt"),
        "acodec": a and a["codec_name"],
        "arate": a and int(a["sample_rate"]),
        "ach": a and a["channels"],
    }


def pcm(path, start, dur):
    """Отрезок звука как float32, моно, SR Гц."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.6f}",
                        "-t", f"{dur:.6f}", "-i", path, "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32)


def hms(s):
    return f"{int(s//3600)}:{int(s%3600//60):02d}:{s%60:06.3f}"


# ------------------------------------------------------- анализ по звуку

def find_overlap(tail, head):
    """
    Насколько начало head повторяет конец tail.
    Нормированная взаимная корреляция через БПФ по всем величинам наложения.
    -> (перехлёст в секундах, NCC)
    """
    La, Lb = len(tail), len(head)
    if La < SR or Lb < SR:
        return 0.0, 0.0
    u = tail - tail.mean()
    v = head - head.mean()
    N = 1 << (La + Lb).bit_length()
    corr = np.fft.irfft(np.fft.rfft(u, N) * np.conj(np.fft.rfft(v, N)), N)[:La]
    eu = np.concatenate(([0.0], np.cumsum(u.astype(np.float64) ** 2)))
    ev = np.concatenate(([0.0], np.cumsum(v.astype(np.float64) ** 2)))
    ov = np.minimum(La - np.arange(La), Lb)          # длина наложения для каждого сдвига
    ncc = corr / (np.sqrt((eu[La] - eu[La - ov]) * ev[ov]) + 1e-9)
    ncc = np.where(ov >= max(int(MIN_OVERLAP * SR), SR // 2), ncc, -1.0)
    k = int(np.argmax(ncc))
    return min(La - k, Lb) / SR, float(ncc[k])


def confirm_overlap(path_a, path_b, ov, dur_a):
    """
    Независимая перепроверка найденного перехлёста.

    Если перехлёст настоящий, то A[dur_a - ov + t] и B[t] — это буквально
    одно и то же место записи, и короткие окна должны совпадать при нулевом
    сдвиге. Ложное срабатывание такую проверку не проходит.

    -> список совпадений по контрольным окнам
    """
    scores = []
    win = max(0.8, min(4.0, ov / 5))
    for frac in CONFIRM_PROBES:
        t = ov * frac
        if t + win > ov:
            continue
        a = pcm(path_a, dur_a - ov + t, win)
        b = pcm(path_b, t, win)
        n = min(len(a), len(b))
        if n < SR // 2:
            continue
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        scores.append(float(np.dot(a, b) /
                            (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return scores


def locate(haystack, needle):
    """Позиция needle внутри haystack. -> (индекс сэмпла, NCC)."""
    Lh, Ln = len(haystack), len(needle)
    h = haystack - haystack.mean()
    n = needle - needle.mean()
    N = 1 << (Lh + Ln).bit_length()
    corr = np.fft.irfft(np.fft.rfft(h, N) * np.conj(np.fft.rfft(n, N)), N)[:Lh - Ln + 1]
    eh = np.concatenate(([0.0], np.cumsum(h.astype(np.float64) ** 2)))
    win = eh[Ln:Lh + 1] - eh[:Lh - Ln + 1]
    ncc = corr / (np.sqrt(win * float(np.dot(n, n))) + 1e-9)
    k = int(np.argmax(ncc))
    return k, float(ncc[k])


def keyframes(path, upto):
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
                "-of", "csv=p=0", "-read_intervals", f"%+{upto:.3f}", path]).stdout
    ks = []
    for line in out.splitlines():
        t = line.strip().rstrip(",")
        if t:
            try:
                ks.append(float(t))
            except ValueError:
                pass
    return sorted(ks)


def order_key(path):
    """Порядок по метке времени в имени: ..._2026-07-30 15-25-42_720p_..."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})",
                  os.path.basename(path))
    return tuple(int(x) for x in m.groups()) if m else (0,) * 6


# ------------------------------------------------------------------ main

def glue(folder, out_name="full_video.mp4", dry_run=False):
    out_path = os.path.join(folder, out_name)
    files = sorted([f for f in glob.glob(os.path.join(folder, "*.mp4"))
                    if os.path.abspath(f) != os.path.abspath(out_path)],
                   key=order_key)
    # уже обработано: результат есть, исходных кусков не осталось
    if not files and os.path.exists(out_path):
        meta_p = os.path.join(folder, "offsets.json")
        if os.path.exists(meta_p):
            print(f"=== УЖЕ ГОТОВО ===\n  {out_name} на месте, offsets.json есть")
            return json.load(open(meta_p))
        d = probe(out_path)
        print(f"=== УЖЕ ГОТОВО ===\n  {out_name} на месте, offsets.json восстановлен")
        meta = {"output": out_name, "duration": d["dur"], "nframes": d["nframes"],
                "fps": d["fps"],
                "sources": [{"file": out_name, "skip": 0.0, "offset": 0.0,
                             "dur": d["dur"], "nframes": d["nframes"]}]}
        with open(meta_p, "w") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        return meta

    if not files:
        print("в папке нет mp4")
        return None
    info = {f: probe(f) for f in files}

    # один кусок — склеивать нечего, просто приводим к общему имени
    if len(files) == 1:
        src = files[0]
        d = info[src]
        print(f"=== ОДИН КУСОК, СКЛЕЙКА НЕ НУЖНА ===\n  {os.path.basename(src)[:76]}")
        if not dry_run:
            os.rename(src, out_path)
            meta = {"output": os.path.basename(out_path), "duration": d["dur"],
                    "nframes": d["nframes"], "fps": d["fps"],
                    "sources": [{"file": os.path.basename(src), "skip": 0.0,
                                 "offset": 0.0, "dur": d["dur"],
                                 "nframes": d["nframes"]}]}
            with open(os.path.join(folder, "offsets.json"), "w") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=1)
            print(f"  переименован в {out_name}, смещение 0")
            return meta
        return {"files": files, "snapped": [0.0]}

    print("=== 1. ПОРЯДОК ===")
    for i, f in enumerate(files, 1):
        d = info[f]
        print(f"  {i}. {d['dur']:9.3f}с {d['nframes']:>7}к  {os.path.basename(f)[:70]}")

    sig = {(d["w"], d["h"], round(d["fps"], 3), d["vcodec"], d["pix"],
            d["acodec"], d["arate"], d["ach"]) for d in info.values()}
    if len(sig) != 1:
        print("\nОШИБКА: параметры кусков различаются, стрим-копирование невозможно.")
        for f, d in info.items():
            print(f"  {d['w']}x{d['h']}@{d['fps']:.3f} {d['vcodec']}/{d['pix']} "
                  f"{d['acodec']} {d['arate']}Hz {d['ach']}ch  {os.path.basename(f)[:50]}")
        return None
    print("  параметры всех кусков совпадают")

    print("\n=== 2. ПЕРЕХЛЁСТЫ ПО ЗВУКУ ===")
    cuts = [0.0]
    for i in range(len(files) - 1):
        A, B = files[i], files[i + 1]
        dA = info[A]["dur"]
        ov, ncc = find_overlap(pcm(A, dA - min(WIN, dA), min(WIN, dA)),
                               pcm(B, 0.0, min(WIN, info[B]["dur"])))
        if ncc < NCC_MIN or ov < MIN_OVERLAP:
            print(f"  стык {i+1}->{i+2}: перехлёста нет (NCC={ncc:.2f})")
            cuts.append(0.0)
            continue
        # первичной корреляции мало — подтверждаем на независимых окнах
        sc = confirm_overlap(A, B, ov, dA)
        worst = min(sc) if sc else 0.0
        det = " ".join(f"{s:.2f}" for s in sc)
        if len(sc) >= 2 and worst >= CONFIRM_MIN:
            print(f"  стык {i+1}->{i+2}: перехлёст {ov:.3f} с "
                  f"(NCC={ncc:.2f}, подтверждение [{det}])")
            cuts.append(ov)
        else:
            print(f"  стык {i+1}->{i+2}: ОТКЛОНЁН перехлёст {ov:.3f} с — "
                  f"не подтвердился (NCC={ncc:.2f}, окна [{det}], худшее {worst:.2f})")
            cuts.append(0.0)

    print("\n=== 3. РЕЗЫ НА КЛЮЧЕВЫЕ КАДРЫ ===")
    snapped = []
    for f, c in zip(files, cuts):
        if c <= 0:
            snapped.append(0.0)
            continue
        ks = keyframes(f, c + 6.0)
        best = min(ks, key=lambda k: abs(k - c)) if ks else c
        fr = 1.0 / info[f]["fps"]
        print(f"  {os.path.basename(f)[:46]}: {c:.3f} -> {best:.3f} "
              f"(промах {abs(best-c)*1000:.0f} мс = {abs(best-c)/fr:.2f} кадра)")
        snapped.append(best)

    if dry_run:
        print("\n--dry-run: файл не создаётся")
        return {"files": files, "snapped": snapped}

    print("\n=== 4. СКЛЕЙКА (стрим-копирование) ===")
    lst = os.path.join(folder, ".concat.txt")
    with open(lst, "w") as fh:
        for f, s in zip(files, snapped):
            fh.write(f"file '{os.path.abspath(f)}'\n")
            if s > 0:
                fh.write(f"inpoint {s:.6f}\n")
    r = _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", lst, "-c", "copy", "-movflags", "+faststart", out_path])
    os.unlink(lst)
    if r.returncode != 0:
        print("ОШИБКА ffmpeg:", r.stderr[:500])
        return None
    o = probe(out_path)
    exp = sum(info[f]["nframes"] - round(s * info[f]["fps"])
              for f, s in zip(files, snapped))
    print(f"  {hms(o['dur'])} | кадров {o['nframes']} (ждали {exp}, "
          f"разница {o['nframes']-exp}) | {os.path.getsize(out_path)/2**30:.2f} ГиБ")

    print("\n=== 5. ЗАМЕР ФАКТИЧЕСКИХ СМЕЩЕНИЙ ===")
    pred, t = [], 0.0
    for f, s in zip(files, snapped):
        pred.append(t)
        t += info[f]["dur"] - s
    offsets = []
    for i, (f, s) in enumerate(zip(files, snapped)):
        ref = s + min(120.0, (info[f]["dur"] - s) / 3)
        k, ncc = locate(pcm(out_path, pred[i] + ref - s - 4.0, 28.0), pcm(f, ref, 20.0))
        off = pred[i] + ref - s - 4.0 + k / SR - ref
        offsets.append(off)
        flag = "" if ncc > 0.95 else "  <-- НИЗКАЯ УВЕРЕННОСТЬ"
        print(f"  кусок {i+1}: смещение {off:11.4f} с  NCC={ncc:.4f}{flag}")

    meta = {
        "output": os.path.basename(out_path),
        "duration": o["dur"], "nframes": o["nframes"], "fps": o["fps"],
        "sources": [{"file": os.path.basename(f), "skip": s, "offset": off,
                     "dur": info[f]["dur"], "nframes": info[f]["nframes"]}
                    for f, s, off in zip(files, snapped, offsets)],
    }
    with open(os.path.join(folder, "offsets.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"\n  готово: {out_path}")
    print(f"  смещения: {os.path.join(folder, 'offsets.json')}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Склейка кусков стрима без перекодирования")
    ap.add_argument("folder", help="папка с mp4-кусками")
    ap.add_argument("-o", "--output", default="full_video.mp4")
    ap.add_argument("--dry-run", action="store_true", help="только анализ, без склейки")
    a = ap.parse_args()
    sys.exit(0 if glue(a.folder, a.output, a.dry_run) else 1)
