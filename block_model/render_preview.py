#!/usr/bin/env python3
"""
Рендер куска готовой нарезки — чтобы посмотреть глазами, не собирая час целиком.

Берёт монтажный лист, отбирает отрезки, которые покрывают заданный участок
ВЫПУСКА (а не стрима), и склеивает. Кодирование через videotoolbox, поэтому
пять минут собираются за минуту-другую и машина не греется.

  render_preview.py --edl data/edl_s2_final.json --from 0 --minutes 5 --out начало
"""
import json, os, subprocess, argparse, tempfile, shutil
import concurrent.futures as cf

BASE = os.environ.get("SEM_STREAMS", "work/streams")
OUT = os.environ.get("SEM_RENDER", os.path.expanduser("~/Downloads"))


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edl", required=True)
    ap.add_argument("--from", dest="frm", type=float, default=0.0,
                    help="с какой минуты ВЫПУСКА начать")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out", default="кусок")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--bitrate", default="3M")
    a = ap.parse_args()

    d = json.load(open(a.edl))
    slot = d["slot"]
    video = f"{BASE}/{slot}/streams/full_video.mp4"
    segs = d["segments"]

    # отбираем отрезки, попадающие в нужный участок выпуска
    acc, chosen = 0.0, []
    lo, hi = a.frm * 60, (a.frm + a.minutes) * 60
    for s, e in segs:
        dur = e - s
        if acc + dur > lo and acc < hi:
            cut_s = s + max(0.0, lo - acc)
            cut_e = s + min(dur, hi - acc)
            if cut_e - cut_s > 0.4:
                chosen.append([cut_s, cut_e])
        acc += dur
        if acc >= hi:
            break
    total = sum(e - s for s, e in chosen)
    print(f"слот {slot}: {len(chosen)} кусков, {total/60:.1f} мин "
          f"(участок выпуска {a.frm:.0f}–{a.frm+a.minutes:.0f} мин)")

    tmp = tempfile.mkdtemp(prefix="part_")

    def one(ia):
        i, (s, e) = ia
        p = f"{tmp}/p{i:04d}.mp4"
        r = sh(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{s:.3f}",
                "-i", video, "-t", f"{e-s:.3f}",
                "-vf", f"scale={a.width}:-2",
                "-c:v", "h264_videotoolbox", "-b:v", a.bitrate,
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
                "-avoid_negative_ts", "make_zero", "-y", p])
        return p if r.returncode == 0 and os.path.exists(p) else None

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        parts = list(ex.map(one, enumerate(chosen)))
    ok = [p for p in parts if p]
    lst = f"{tmp}/list.txt"
    with open(lst, "w") as f:
        for p in ok:
            f.write(f"file '{p}'\n")
    out_path = f"{OUT}/{a.out}.mp4"
    r = sh(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
            "-c", "copy", "-movflags", "+faststart", "-y", out_path])
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        print("ОШИБКА склейки:", r.stderr[-300:]); return
    size = os.path.getsize(out_path) / 2 ** 20
    dur = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", out_path]).stdout.strip()
    print(f"готово: {out_path} — {float(dur)/60:.1f} мин, {size:.0f} МБ, "
          f"нарезано {len(ok)}/{len(chosen)}")


if __name__ == "__main__":
    main()
