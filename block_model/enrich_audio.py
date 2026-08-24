#!/usr/bin/env python3
"""
Звуковые пометки к блокам: то, чего нет в расшифровке.

Расшифровка не передаёт, как это звучало. Смех, крик, гул толпы, шум машины
(значит едем), громкость относительно самого стрима, темп речи — всё это
монтажёр слышит, а модель по тексту не видит.

Берём готовые теги CLAP (шаг 5 с, 16 тегов) и громкость из ленты элементов.
Пороги — не абсолютные, а по самому стриму: у каждого эфира своя громкость
и свой микрофон.

  enrich_audio.py --slot 5
Дописывает поле "au" в data/s<N>.blocks.json.
"""
import json, argparse, os
import numpy as np

DATASET = os.environ.get("SEM_DATASET", "work/dataset")
DATA = os.environ.get("SEM_BLOCKS", "work/blocks")
SRC = {
    "1": "slot1_2026-07-30_futbol-mozgi.labeled.json",
    "2": "slot2_2026-07-31_gulyanka.json",
    "3": "slot3_2026-08-03_tarzanka-zabroshka.labeled.json",
    "4": "slot4_2026-08-04_razvlekalovka.labeled.json",
    "5": "slot5_2026-06-12_matiz.labeled.json",
    "6": "slot6_2026-06-30_2mp4.labeled.json",
    "7": "slot7_2026-06-30_varenikovskiy.labeled.json",
}
# Показываем только теги, у которых замерена связь с решением монтажёра
# (check_audio.py, порог 5 %, отклонение доли оставленного от базы 39,3 %):
#   смех +16, машина −23, телефон −18, тишина −15, музыка −14, улица −14,
#   еда −13, ветер −11, вода −20.
# Выброшены как пустые: крик +2, аплодисменты −4, стук −0,5, толпа, «один».
SHOW = {"смех": "смех", "машина": "едут в машине", "телефон": "звук из телефона",
        "музыка": "играет музыка", "еда": "кафе, звон посуды",
        "тишина": "почти тихо", "улица": "уличный шум",
        "ветер": "ветер в микрофон", "вода": "ветер в микрофон"}
Q = 0.95            # тег показываем, если он в верхних 5 % этого стрима
LOUD_Q, QUIET_Q = 0.80, 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    a = ap.parse_args()
    s = a.slot
    B = json.load(open(f"{DATA}/s{s}.blocks.json"))
    blocks = B["blocks"]
    items = json.load(open(f"{DATASET}/{SRC[s]}"))["items"]

    st = np.array([r["start"] for r in items])
    en = np.array([r["end"] for r in items])
    vol = np.array([r["f"]["vol"] for r in items])
    isw = np.array([r["f"]["is_word"] for r in items])

    cl = np.load(f"{DATA}/clap/clap_s{s}.npz", allow_pickle=True)
    C = cl["clap"].astype(np.float32)
    names = [str(x) for x in cl["names"]]
    hop = float(cl["clap_hop"])
    thr = {n: np.quantile(C[:, j], Q) for j, n in enumerate(names)}

    # громкость блока — по речевым элементам внутри него
    bvol = []
    for b in blocks:
        m = (st < b["end"]) & (en > b["start"]) & (isw == 1)
        bvol.append(float(vol[m].mean()) if m.any() else np.nan)
    bvol = np.array(bvol)
    good = ~np.isnan(bvol)
    lo_v = np.quantile(bvol[good], QUIET_Q) if good.any() else 0
    hi_v = np.quantile(bvol[good], LOUD_Q) if good.any() else 1

    # темп речи: слов в минуту относительно стрима
    wpm = []
    for b in blocks:
        m = (st < b["end"]) & (en > b["start"]) & (isw == 1)
        wpm.append(m.sum() / max(b["dur"], 0.1) * 60)
    wpm = np.array(wpm)
    hi_w = np.quantile(wpm[wpm > 0], 0.85) if (wpm > 0).any() else 1e9

    for k, b in enumerate(blocks):
        tags = []
        i0 = int(b["start"] / hop); i1 = max(i0 + 1, int(np.ceil(b["end"] / hop)))
        i1 = min(i1, len(C))
        if i1 > i0:
            seg = C[i0:i1]
            for j, n in enumerate(names):
                if n in SHOW and seg[:, j].max() >= thr[n]:
                    tags.append(SHOW[n])
        if b["kind"] == "speech" and good[k]:
            if bvol[k] >= hi_v:
                tags.append("громко")
            elif bvol[k] <= lo_v:
                tags.append("тихо")
            if wpm[k] >= hi_w:
                tags.append("тараторят")
        b["au"] = tags

    json.dump(B, open(f"{DATA}/s{s}.blocks.json", "w"), ensure_ascii=False)
    n = sum(1 for b in blocks if b["au"])
    from collections import Counter
    c = Counter(t for b in blocks for t in b["au"])
    print(f"слот {s}: пометки у {n}/{len(blocks)} блоков — "
          + ", ".join(f"{k} {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
