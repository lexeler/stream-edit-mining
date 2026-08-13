#!/usr/bin/env python3
"""
Пересчёт монтажного листа с отдельных кусков на склеенный файл.

    python3 remap.py <папка> <проект.prproj> [--no-check]

Читает .prproj (это gzip-XML), достаёт список отрезков «оставить»,
и переводит их таймкоды в систему координат full_video.mp4,
пользуясь смещениями из offsets.json (их пишет glue.py).

На выходе timecodes.txt и timecodes.json.

Оговорка: переносится только раскладка отрезков. Эффекты, маски,
размытия и переходы в лист не попадают — их в нём просто нет.
"""
import argparse, glob, gzip, json, os, re, subprocess, sys
import xml.etree.ElementTree as ET
import numpy as np

TICKS = 254016000000.0    # тиков в секунде в формате Premiere


def gray(path, t, w=192, h=108):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.6f}", "-i", path,
                        "-frames:v", "1", "-vf", f"scale={w}:{h},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    return a[:w * h].astype(np.float32) if a.size >= w * h else None


def hms(s):
    return f"{int(s//3600)}:{int(s%3600//60):02d}:{s%60:06.3f}"


def parse_prproj(path):
    """-> список отрезков: файл-источник, in/out в нём, позиция на таймлайне."""
    root = ET.parse(gzip.open(path)).getroot()
    by = {e.get("ObjectID"): e for e in root.iter() if e.get("ObjectID")}

    def deref(el, tag):
        c = el.find(tag)
        return by.get(c.get("ObjectRef")) if c is not None and c.get("ObjectRef") else None

    rows = []
    # берём только видеодорожку: аудиоклипы дублируют её один в один
    for vti in root.iter("VideoClipTrackItem"):
        cti = vti.find("ClipTrackItem")
        ti = cti.find("TrackItem") if cti is not None else None
        sub = deref(cti, "SubClip") if cti is not None else None
        if ti is None or sub is None:
            continue
        clip = deref(sub, "Clip")
        a = clip.findtext(".//InPoint") if clip is not None else None
        b = clip.findtext(".//OutPoint") if clip is not None else None
        if a is None or b is None:
            continue
        rows.append({"tl": int(ti.findtext("Start")) / TICKS,
                     "tl_end": int(ti.findtext("End")) / TICKS,
                     "in": int(a) / TICKS, "out": int(b) / TICKS,
                     "file": os.path.basename(sub.findtext("Name") or "?")})
    rows.sort(key=lambda r: r["tl"])
    return rows


def remap(folder, prproj, check=True):
    meta = json.load(open(os.path.join(folder, "offsets.json")))
    out = os.path.join(folder, meta["output"])
    off = {s["file"]: s for s in meta["sources"]}

    rows = parse_prproj(prproj)
    print(f"=== ПРОЕКТ ===\n  отрезков: {len(rows)}")
    unknown = {r["file"] for r in rows} - set(off)
    if unknown:
        print("  ОШИБКА, нет таких исходников в offsets.json:")
        for u in list(unknown)[:5]:
            print("   ", u[:80])
        return None

    print("\n=== ПЕРЕСЧЁТ ===")
    segs, dropped = [], 0
    for r in rows:
        s = off[r["file"]]
        if r["in"] < s["skip"] - 1e-6:
            dropped += 1          # отрезок попал в вырезанный перехлёст
            continue
        segs.append({"start": r["in"] + s["offset"], "end": r["out"] + s["offset"],
                     "dur": r["out"] - r["in"], "src": r["file"], "src_in": r["in"]})
    total = sum(x["dur"] for x in segs)
    tl = rows[-1]["tl_end"]
    print(f"  перенесено: {len(segs)}" + (f", ПОТЕРЯНО {dropped}" if dropped else ""))
    print(f"  сумма отрезков:  {hms(total)}")
    print(f"  таймлайн проекта: {hms(tl)}")
    print(f"  расхождение: {abs(total - tl)*1000:.1f} мс")
    mono = all(segs[i]["start"] >= segs[i - 1]["start"] for i in range(1, len(segs)))
    print(f"  порядок по возрастанию: {'да' if mono else 'НЕТ'}")

    with open(os.path.join(folder, "timecodes.json"), "w") as fh:
        json.dump(segs, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(folder, "timecodes.txt"), "w") as fh:
        fh.write(f"# нарезка для {meta['output']}\n"
                 f"# отрезков: {len(segs)}, итог: {hms(total)}\n"
                 f"# начало\tконец\tдлина_с\n")
        for x in segs:
            fh.write(f"{hms(x['start'])}\t{hms(x['end'])}\t{x['dur']:.3f}\n")
    print("  сохранено: timecodes.txt, timecodes.json")

    if check and segs:
        print("\n=== ПРОВЕРКА: кадр по новому таймкоду == кадр из исходника ===")
        import unicodedata as u
        nfc = lambda s: u.normalize("NFC", s)
        srcs = {nfc(os.path.basename(f)): f
                for f in glob.glob(os.path.join(folder, "*.mp4"))}
        # единственный кусок переименован в результат — сверять не с чем и незачем
        if len(meta["sources"]) == 1 and nfc(meta["sources"][0]["file"]) not in srcs:
            print("  слот из одного куска: исходник и есть результат, сверка не нужна")
            return segs
        bad = skipped = 0
        for j in np.linspace(0, len(segs) - 1, 8).astype(int):
            x = segs[j]
            src = srcs.get(nfc(x["src"]))
            if src is None:
                skipped += 1
                continue
            d = min(1.0, x["dur"] / 2)
            g1, g2 = gray(out, x["start"] + d), gray(src, x["src_in"] + d)
            if g1 is None or g2 is None:
                skipped += 1
                continue
            diff = float(np.mean(np.abs(g1 - g2)))
            bad += 0 if diff < 2.0 else 1
            print(f"  отрезок {j:>4} @ {hms(x['start'])}: разница {diff:5.2f} "
                  f"{'OK' if diff < 2.0 else 'НЕ СОВПАЛО'}")
        note = f", пропущено {skipped}" if skipped else ""
        print(f"\n  {'все проверенные кадры совпали' if bad == 0 else f'{bad} расхождений'}{note}")
    return segs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Пересчёт монтажного листа на склеенный файл")
    ap.add_argument("folder")
    ap.add_argument("prproj")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if remap(a.folder, a.prproj, not a.no_check) else 1)
