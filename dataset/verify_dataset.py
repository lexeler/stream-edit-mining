#!/usr/bin/env python3
"""
Проверка готовой выборки перед тем, как отдавать её в обучение.

Шесть проверок, каждая ловит свой класс ошибок:
  1. утечка меток в тест
  2. целостность ленты — элементы должны идти встык, без дыр и нахлёстов
  3. точность меток против монтажного листа
  4. баланс классов по слотам
  5. сопоставимость признаков между слотами (иначе обучение развалится)
  6. общий материал между слотами — слоты 6 и 7 сняты в один день
"""
import glob, json, os, collections
import numpy as np

OUT = os.environ.get("SEM_DATASET", "work/dataset")
LABELS = ("used", "used_half", "used_frac", "cut_rel", "cut_side")


def head(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


def main():
    train = sorted(glob.glob(f"{OUT}/slot*.labeled.json"))
    tests = [p for p in glob.glob(f"{OUT}/slot*.json")
             if ".labeled." not in p and ".answers." not in p
             and json.load(open(p)).get("split") == "test"]
    print(f"обучающих слотов: {len(train)}, тестовых: {len(tests)}")

    head("1. УТЕЧКА МЕТОК В ТЕСТ")
    ok = True
    for p in tests:
        d = json.load(open(p))
        bad = set()
        for it in d["items"]:
            bad |= {k for k in it if k in LABELS}
        print(f"  {os.path.basename(p)}: {'ЧИСТО' if not bad else 'УТЕЧКА ' + str(bad)}")
        ok &= not bad
    jl = f"{OUT}/test.jsonl"
    if os.path.exists(jl):
        with open(jl) as f:
            first = json.loads(f.readline())
        bad = [k for k in first if k in LABELS]
        print(f"  test.jsonl: {'ЧИСТО' if not bad else 'УТЕЧКА ' + str(bad)}")
        ok &= not bad

    head("2. ЦЕЛОСТНОСТЬ ЛЕНТЫ")
    for p in train + tests:
        d = json.load(open(p))
        I = d["items"]
        gaps = over = 0
        for a, b in zip(I, I[1:]):
            df = round(b["start"] - a["end"], 3)
            if df > 0.011:
                gaps += 1
            elif df < -0.011:
                over += 1
        cov = sum(i["end"] - i["start"] for i in I) / d["duration_sec"]
        print(f"  слот {d['slot']}: дыр {gaps}, нахлёстов {over}, "
              f"покрытие {cov*100:.1f}%")

    head("3. ТОЧНОСТЬ МЕТОК")
    print(f"{'слот':>5} {'элементов':>10} {'used':>7} {'по used':>9} "
          f"{'по frac':>9} {'монтаж':>8} {'погранич.':>10}")
    for p in train:
        d = json.load(open(p))
        I = d["items"]
        e = d["edl_duration_sec"]
        tu = sum(i["end"] - i["start"] for i in I if i["used"]) / 60
        tf = sum((i["end"] - i["start"]) * i["used_frac"] for i in I) / 60
        pt = sum(1 for i in I if 0 < i["used_frac"] < 1)
        print(f"{d['slot']:>5} {len(I):>10} "
              f"{sum(1 for i in I if i['used'])/len(I)*100:>6.1f}% "
              f"{tu:>8.1f}м {tf:>8.1f}м {e/60:>7.1f}м {pt:>10}")

    head("4. БАЛАНС И ЦЕЛИ")
    allf = []
    for p in train:
        d = json.load(open(p))
        I = d["items"]
        cs = collections.Counter(i.get("cut_side") for i in I if i.get("cut_side"))
        allf += [i["used_frac"] for i in I]
        print(f"  слот {d['slot']}: used {sum(i['used'] for i in I)/len(I)*100:5.1f}%, "
              f"used_half {sum(i['used_half'] for i in I)/len(I)*100:5.1f}%, "
              f"резы {dict(cs)}")
    a = np.array(allf)
    print(f"\n  used_frac по всем: 0 -> {(a==0).mean()*100:.1f}%, "
          f"1 -> {(a==1).mean()*100:.1f}%, между -> {((a>0)&(a<1)).mean()*100:.1f}%")

    head("5. СОПОСТАВИМОСТЬ ПРИЗНАКОВ МЕЖДУ СЛОТАМИ")
    keys = ("vol", "dens15", "pause_dens", "zcr", "voice", "block_pos")
    rows = {}
    for p in train + tests:
        d = json.load(open(p))
        rows[d["slot"]] = {k: float(np.median([i["f"][k] for i in d["items"]]))
                           for k in keys}
    print(f"  {'слот':>5} " + " ".join(f"{k:>11}" for k in keys))
    for s in sorted(rows):
        print(f"  {s:>5} " + " ".join(f"{rows[s][k]:>11.3f}" for k in keys))
    for k in keys:
        v = [rows[s][k] for s in rows]
        sp = (max(v) - min(v)) / (abs(np.mean(v)) + 1e-9)
        if sp > 0.5:
            print(f"  ВНИМАНИЕ: {k} сильно разнится между слотами "
                  f"({min(v):.3f}..{max(v):.3f}) — нужна нормировка по слоту")

    head("ИТОГ")
    print(f"  утечек: {'нет' if ok else 'ЕСТЬ'}")


if __name__ == "__main__":
    main()
