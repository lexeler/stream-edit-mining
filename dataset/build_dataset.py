#!/usr/bin/env python3
"""
Сборка итоговых файлов для обучения.

train.jsonl        — по строке на элемент, все обучающие слоты
test.jsonl         — тестовый слот, БЕЗ меток
test_answers.jsonl — ответы к нему отдельно
"""
import json, os, glob, collections

OUT = os.environ.get("SEM_DATASET", "work/dataset")
LABEL_KEYS = ("used", "used_half", "used_frac", "cut_rel", "cut_side")


def rows(path, with_labels):
    d = json.load(open(path))
    slot = d["slot"]
    for k, it in enumerate(d["items"]):
        r = {"slot": slot, "i": k, "type": it["type"],
             "start": it["start"], "end": it["end"]}
        if it["type"] == "word":
            r["word"] = it["word"]
        r.update(it["f"])
        if with_labels:
            for key in LABEL_KEYS:
                if key in it:
                    r[key] = it[key]
        yield r


def main():
    train, test, answers = [], [], []
    stats = []
    for p in sorted(glob.glob(f"{OUT}/slot*.labeled.json")):
        n = 0
        for r in rows(p, True):
            train.append(r)
            n += 1
        d = json.load(open(p))
        stats.append((d["slot"], "train", n, d["duration_sec"],
                      d.get("edl_duration_sec", 0)))

    for p in sorted(glob.glob(f"{OUT}/slot*.json")):
        if ".labeled." in p or ".answers." in p:
            continue
        d = json.load(open(p))
        if d.get("split") != "test":
            continue
        for r in rows(p, False):
            test.append(r)
        a = json.load(open(p.replace(".json", ".answers.json")))
        answers.extend(a["answers"])
        stats.append((d["slot"], "test", len(test), d["duration_sec"],
                      a.get("edl_duration_sec", 0)))

    def dump(name, data):
        with open(f"{OUT}/{name}", "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return os.path.getsize(f"{OUT}/{name}") / 2 ** 20

    s1 = dump("train.jsonl", train)
    s2 = dump("test.jsonl", test)
    s3 = dump("test_answers.jsonl", answers)

    print(f"{'слот':>5} {'набор':>7} {'строк':>9} {'часов':>7} {'монтаж,мин':>11}")
    for sl, sp, n, du, ed in sorted(stats):
        print(f"{sl:>5} {sp:>7} {n:>9} {du/3600:>7.2f} {ed/60:>11.1f}")
    print()
    print(f"train.jsonl        {len(train):>7} строк  {s1:>6.1f} МБ")
    print(f"test.jsonl         {len(test):>7} строк  {s2:>6.1f} МБ")
    print(f"test_answers.jsonl {len(answers):>7} строк  {s3:>6.1f} МБ")

    if train:
        u = sum(1 for r in train if r.get("used"))
        uf = sum(r.get("used_frac", 0) * (r["end"] - r["start"]) for r in train)
        tot = sum(r["end"] - r["start"] for r in train)
        print(f"\nобучение: used {u/len(train)*100:.1f}% элементов, "
              f"по времени {uf/tot*100:.1f}%")
        part = sum(1 for r in train if 0 < r.get("used_frac", 0) < 1)
        print(f"пограничных (0<used_frac<1): {part} = {part/len(train)*100:.1f}%")
        cs = collections.Counter(r.get("cut_side") for r in train if r.get("cut_side"))
        print(f"из них: {dict(cs)}")
    # проверка на утечку: в тесте не должно быть ни одного поля метки
    bad = [k for k in (test[0] if test else {}) if k in LABEL_KEYS]
    print(f"\nутечка меток в test.jsonl: {'НАЙДЕНА ' + str(bad) if bad else 'нет'}")


if __name__ == "__main__":
    main()
