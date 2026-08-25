#!/usr/bin/env python3
"""
Проект Premiere с нашей нарезкой — чтобы монтажёр открыл и увидел резы.

Собирать .prproj с нуля рискованно: внутри сотни объектов, связанных
перекрёстными ссылками ObjectID/ObjectRef. Поэтому берём настоящий проект
этого же монтажёра как основу и правим в нём ровно три вещи:

  * тайминги клипов — Start/End (позиция на таймлайне) и InPoint/OutPoint
    (позиция в исходнике), единица времени 254016000000 тиков в секунду;
  * длину списка клипов на дорожках — лишние убираем;
  * путь к исходнику — на обрезанный кусок стрима.

Эффекты (замазывание баннеров и прочее) не переносятся: нужна только
нарезка.

  ./export_prproj.py --slot 2 --edl data/edl_s2_first20.json --media r20.mp4
"""
import gzip, json, os, re, argparse

TICKS = 254016000000
DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
# проект монтажёра, на основе которого собирается новый (по слоту)
TEMPLATES = {"2": os.environ.get("SEM_PRPROJ_TEMPLATE", "work/projects/2/project.prproj")}


def cut_list(xml, tag, keep):
    """оставляем в списке клипов дорожки только первые keep штук"""
    def fix(m):
        body = m.group(0)
        items = re.findall(r'<TrackItem Index="\d+" ObjectRef="\d+"/>', body)
        if len(items) <= keep:
            return body
        head = body[:body.find(items[0])]
        tail = body[body.find(items[-1]) + len(items[-1]):]
        new = "\n".join(
            re.sub(r'Index="\d+"', f'Index="{i}"', it)
            for i, it in enumerate(items[:keep]))
        return head + new + tail
    return re.sub(r'<TrackItems Version="1">.*?</TrackItems>', fix, xml,
                  flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="2")
    ap.add_argument("--edl", required=True)
    ap.add_argument("--media", required=True, help="имя файла-исходника")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    segs = json.load(open(a.edl))["наша"]
    n = len(segs)
    xml = gzip.open(TEMPLATES[a.slot], "rb").read().decode("utf-8")

    vids = re.findall(r'<VideoClipTrackItem ObjectID="(\d+)".*?'
                      r'<Start>(-?\d+)</Start>\s*<End>(-?\d+)</End>.*?'
                      r'<SubClip ObjectRef="(\d+)"/>', xml, re.S)
    auds = re.findall(r'<AudioClipTrackItem ObjectID="(\d+)".*?'
                      r'<Start>(-?\d+)</Start>\s*<End>(-?\d+)</End>.*?'
                      r'<SubClip ObjectRef="(\d+)"/>', xml, re.S)
    print(f"в шаблоне клипов: видео {len(vids)}, аудио {len(auds)}; "
          f"нужно {n}")
    if n > min(len(vids), len(auds)):
        print("в шаблоне меньше клипов, чем в нарезке — так нельзя")
        return

    def subclip_to_clip(sub_id):
        m = re.search(rf'<SubClip ObjectID="{sub_id}".*?<Clip ObjectRef="(\d+)"/>',
                      xml, re.S)
        return m.group(1) if m else None

    # новые тайминги
    pos = 0.0
    plan = []
    for i, (s, e) in enumerate(segs):
        dur = e - s
        plan.append((int(round(pos * TICKS)), int(round((pos + dur) * TICKS)),
                     int(round(s * TICKS)), int(round(e * TICKS))))
        pos += dur

    def apply(items, tag):
        nonlocal xml
        for i, (oid, st, en, sub) in enumerate(items[:n]):
            ns, ne, ip, op = plan[i]
            # позиция на таймлайне
            pat = (rf'(<{tag} ObjectID="{oid}".*?<Start>)-?\d+(</Start>\s*<End>)'
                   rf'-?\d+(</End>)')
            xml = re.sub(pat, rf'\g<1>{ns}\g<2>{ne}\g<3>', xml, count=1, flags=re.S)
            # позиция в исходнике
            cid = subclip_to_clip(sub)
            if cid:
                pat2 = (rf'(ObjectID="{cid}".*?<InPoint>)-?\d+(</InPoint>\s*'
                        rf'<OutPoint>)-?\d+(</OutPoint>)')
                xml = re.sub(pat2, rf'\g<1>{ip}\g<2>{op}\g<3>', xml,
                             count=1, flags=re.S)

    apply(vids, "VideoClipTrackItem")
    apply(auds, "AudioClipTrackItem")
    xml = cut_list(xml, "TrackItems", n)

    # исходник
    xml = re.sub(r'<FilePath>[^<]*</FilePath>',
                 f'<FilePath>{a.media}</FilePath>', xml)
    xml = re.sub(r'<RelativePath>[^<]*</RelativePath>',
                 f'<RelativePath>{a.media}</RelativePath>', xml)
    xml = re.sub(r'<Title>[^<]*</Title>', f'<Title>{a.media}</Title>', xml)
    xml = re.sub(r'(<Name>)[^<]*\.mp4(</Name>)', rf'\g<1>{a.media}\g<2>', xml)

    with gzip.open(a.out, "wb") as f:
        f.write(xml.encode("utf-8"))
    total = sum(e - s for s, e in segs)
    print(f"проект: {a.out}")
    print(f"  {n} клипов, {total/60:.1f} мин, исходник «{a.media}»")


if __name__ == "__main__":
    main()
