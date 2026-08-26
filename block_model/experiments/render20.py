#!/usr/bin/env python3
"""
Рендер нарезки первых 20 минут: наша и монтажёрская, для сравнения глазами.

Одним фильтром ffmpeg сотни отрезков не тянет, поэтому режем по одному
и склеиваем. Кодирование через h264_videotoolbox — на Маке это аппаратно
и быстро.

  ./render20.py --slot 2
"""
import json, os, argparse, subprocess, tempfile, shutil, time
import concurrent.futures as cf

DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
BASE = os.environ.get("SEM_STREAMS", "work/streams")
OUT = os.environ.get("SEM_RENDER", os.path.expanduser("~/Downloads"))


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def render(video, segs, out_path, label, workers=4):
    tmp = tempfile.mkdtemp(prefix="cut20_")
    total = sum(b - a for a, b in segs)
    print(f"  {label}: {len(segs)} кусков, {total/60:.1f} мин", flush=True)
    t0 = time.time()

    def one(ia):
        i, (a, b) = ia
        p = f"{tmp}/p{i:05d}.mp4"
        r = sh(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{a:.3f}",
                "-i", video, "-t", f"{b-a:.3f}",
                "-vf", "scale=1280:-2",
                "-c:v", "h264_videotoolbox", "-b:v", "3M",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                "-avoid_negative_ts", "make_zero", "-y", p])
        return p if r.returncode == 0 and os.path.exists(p) else None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(one, enumerate(segs)))
    ok = [p for p in parts if p]
    print(f"    нарезано {len(ok)}/{len(segs)}", flush=True)
    lst = f"{tmp}/list.txt"
    with open(lst, "w") as f:
        for p in ok:
            f.write(f"file '{p}'\n")
    r = sh(["ffmpeg", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0",
            "-i", lst, "-c", "copy", "-y", out_path])
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        print(f"    ОШИБКА склейки: {r.stderr[-300:]}")
        return None
    sz = os.path.getsize(out_path) / 2**20
    print(f"    готово за {time.time()-t0:.0f} с: {out_path} ({sz:.0f} МБ)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2")
    a = ap.parse_args()
    d = json.load(open(f"{DATA}/edl_s{a.slot}_first20.json"))
    video = f"{BASE}/{a.slot}/streams/full_video.mp4"
    render(video, d["наша"], f"{OUT}/слот{a.slot}_первые20_МОДЕЛЬ.mp4",
           "наша нарезка")
    render(video, d["монтажёр"], f"{OUT}/слот{a.slot}_первые20_МОНТАЖЁР.mp4",
           "монтажёр")


if __name__ == "__main__":
    main()
