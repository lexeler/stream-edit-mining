#!/usr/bin/env python3
"""
Приводит слот к единому виду.

    python3 process_slot.py <папка_слота> [--clean] [--dry-run]

Что получается на выходе:

    <слот>/
    ├── streams/
    │   └── full_video.mp4        один ролик: склеенный или единственный
    └── narezki/
        ├── timecodes.json        для машины
        ├── timecodes.txt         для человека
        └── *.prproj              исходный проект, остаётся как первоисточник

--clean убирает из streams/ всё, кроме full_video.mp4: исходные куски и zip.
Делать только после того, как есть резервная копия.
"""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glue as G
import remap as R
import verify as V


def hms(s):
    return f"{int(s//3600)}:{int(s%3600//60):02d}:{s%60:06.3f}"


def write_outputs(slot_dir, narezki, meta, segs):
    slot = os.path.basename(slot_dir.rstrip("/"))
    edit = sum(x["dur"] for x in segs)
    src = meta["sources"]

    doc = {
        "video_duration_sec": round(meta["duration"], 3),
        "edit_duration_sec": round(edit, 3),
        "segments_total": len(segs),
        "segments": [{"start": round(x["start"], 3), "end": round(x["end"], 3)}
                     for x in segs],
    }
    with open(os.path.join(narezki, "timecodes.json"), "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(narezki, "timecodes.txt"), "w") as fh:
        fh.write(f"полный ролик : {hms(meta['duration'])}\n")
        fh.write(f"после нарезки: {hms(edit)}\n")
        fh.write(f"отрезков     : {len(segs)}\n\n")
        for x in segs:
            fh.write(f'{hms(x["start"])}  {hms(x["end"])}\n')
    return doc


def process(slot_dir, clean=False, dry_run=False):
    streams = os.path.join(slot_dir, "streams")
    narezki = os.path.join(slot_dir, "narezki")
    slot = os.path.basename(slot_dir.rstrip("/"))
    print("#" * 60)
    print(f"# СЛОТ {slot}")
    print("#" * 60)

    if not os.path.isdir(streams):
        print("  нет папки streams — пропуск")
        return None
    pro = glob.glob(os.path.join(narezki, "*.prproj"))
    if not pro:
        print("  нет .prproj в narezki — пропуск")
        return None

    meta = G.glue(streams, "full_video.mp4", dry_run)
    if not meta or dry_run:
        return None

    # A/V проверка обязательна и только пока исходники ещё на месте
    if len(meta["sources"]) > 1:
        print()
        if not V.verify(streams):
            print("\n  ВНИМАНИЕ: проверка синхронности не прошла.")
            if clean:
                print("  очистка отменена — исходники сохранены для разбора")
                clean = False

    print()
    segs = R.remap(streams, pro[0], check=True)
    if segs is None:
        return None

    doc = write_outputs(slot_dir, narezki, meta, segs)
    for junk in ("timecodes.json", "timecodes.txt", "offsets.json"):
        p = os.path.join(streams, junk)
        if os.path.exists(p):
            os.unlink(p)

    if clean:
        removed = 0
        for f in glob.glob(os.path.join(streams, "*")):
            if os.path.basename(f) == "full_video.mp4":
                continue
            sz = os.path.getsize(f)
            os.unlink(f)
            removed += sz
            print(f"  удалён {os.path.basename(f)[:60]}")
        if removed:
            print(f"  освобождено {removed/2**30:.2f} ГиБ")

    print(f"\n  СЛОТ {slot} ГОТОВ: {doc['segments_total']} отрезков, "
          f"нарезка {hms(doc['edit_duration_sec'])} из {hms(doc['video_duration_sec'])}")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Привести слот к единому виду")
    ap.add_argument("slots", nargs="+", help="папки слотов")
    ap.add_argument("--clean", action="store_true",
                    help="убрать из streams всё, кроме full_video.mp4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    done = []
    for s in a.slots:
        d = process(s, a.clean, a.dry_run)
        if d:
            done.append(d)
        print()
    if done:
        print("=" * 60)
        print(f"{'слот':>5} {'ролик':>13} {'нарезка':>13} {'отрезков':>9} {'кусков':>7}")
        for d in done:
            print(f"{d['slot']:>5} {hms(d['video_duration_sec']):>13} "
                  f"{hms(d['edit_duration_sec']):>13} {d['segments_total']:>9} "
                  f"{len(d['glued_from']):>7}")
