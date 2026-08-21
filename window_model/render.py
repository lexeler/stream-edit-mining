#!/usr/bin/env python3
"""
Рендер нарезки по монтажному листу из train_predict.py.

Одно большое выражение фильтра ffmpeg не тянет сотни отрезков — падает
по памяти. Поэтому режем куски по отдельности и склеиваем.

  render.py edl_s2.json [--also-editor]
"""
import json, os, subprocess, sys, argparse, tempfile, shutil
import concurrent.futures as cf

BASE = os.environ.get("SEM_STREAMS", "work/streams")
OUT = os.environ.get("SEM_RENDER", os.path.expanduser("~/Downloads"))


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def render(video, segs, out_path, label, workers=3):
    tmp = tempfile.mkdtemp(prefix="cut_")
    total = sum(b - a for a, b in segs)
    print(f"  {label}: {len(segs)} кусков, {total/60:.1f} мин", flush=True)

    def one(ia):
        i, (a, b) = ia
        p = f"{tmp}/p{i:05d}.mp4"
        r = sh(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{a:.3f}",
                "-i", video, "-t", f"{b-a:.3f}",
                "-c:v", "h264_videotoolbox", "-b:v", "6M",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1",
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
    r = sh(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
            "-c", "copy", "-movflags", "+faststart", "-y", out_path])
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        print("  ОШИБКА склейки:", r.stderr[-300:]); return
    d = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", out_path]).stdout.strip()
    print(f"  готово: {os.path.basename(out_path)} — {float(d)/60:.1f} мин, "
          f"{os.path.getsize(out_path)/2**30:.2f} ГиБ", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edl")
    ap.add_argument("--also-editor", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.edl))
    slot = d["slot"]
    video = f"{BASE}/{slot}/streams/full_video.mp4"
    print(f"слот {slot}, метрики: {d.get('metrics')}")
    render(video, d["segments"], f"{OUT}/слот{slot}_МОДЕЛЬ.mp4", "модель")
    if a.also_editor:
        real = [[s["start"], s["end"]]
                for s in json.load(open(f"{BASE}/{slot}/narezki/timecodes.json"))["segments"]]
        render(video, real, f"{OUT}/слот{slot}_МОНТАЖЁР.mp4", "монтажёр")


if __name__ == "__main__":
    main()
