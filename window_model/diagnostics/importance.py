#!/usr/bin/env python3
"""
На что смотрит модель. Восстанавливаем имена признаков в том же порядке,
в каком их складывает windows.py, и печатаем важность по группам и поимённо.

Это и проверка на здравый смысл: если наверху окажется что-то вроде
номера окна, значит модель учит конкретные эфиры, а не работу монтажёра.
"""
import sys, collections
import numpy as np
from catboost import CatBoostRegressor
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
from windows import get_data, AGG_MEAN, AGG_WORD, DATA

BASE = ["длина окна", "доля речи", "слов", "слов в минуту", "доля пауз",
        "макс пауза", "пауз>1с", "пауз>3с"]


def ext_names(blocks):
    out = []
    if "pca" in blocks:
        out += [f"e5-близко-{i}" for i in range(48)] + \
               [f"e5-далеко-{i}" for i in range(48)]
    if "tag" in blocks:
        t = np.load(f"{DATA}/semtags.npy", allow_pickle=True).item()
        nm = list(t[list(t)[0]]["names"])
        out += [f"текст:{n}(близко)" for n in nm] + \
               [f"текст:{n}(далеко)" for n in nm]
    if "vid" in blocks:
        import json
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "dataset"))
            from clip_feat import PROMPTS
            vn = [p[:26] for p in PROMPTS]
        except Exception:
            vn = [f"кадр-{i}" for i in range(24)]
        out += [f"кадр:{n}" for n in vn]
        out += ["застой 30с", "застой 2мин", "застой 5мин",
                "яркость", "движение", "смена плана"]
    if "vemb" in blocks:
        out += [f"кадр-вектор-{i}" for i in range(32)]
    if "clap" in blocks:
        a = np.load(f"{DATA}/clap_s1.npz", allow_pickle=True)
        out += [f"звук:{n}" for n in list(a["names"])]
    if "xv" in blocks:
        out += ["голос: далеко от хозяина", "голос: смена",
                "голос 15с", "голос 60с", "голос 5мин"]
    if "llm" in blocks:
        a = np.load(f"{DATA}/llm_s1.npz", allow_pickle=True)
        out += [f"Qwen:{n}" for n in list(a["names"])]
    return out


def all_names(blocks):
    base = BASE + sum([[f"{k}", f"{k} разброс"] for k in AGG_MEAN], []) + \
        [f"слово:{k}" for k in AGG_WORD]
    nac = len(base)
    full = base + ext_names(blocks)
    m = len(full)
    names = [f"{n}" for n in full] + [f"{n} (норм)" for n in full]
    for lag in (1, 2, 4, 8, 16):
        names += [f"{n} (−{lag})" for n in base]
        names += [f"{n} (+{lag})" for n in base]
    names += [f"{n} (мин 1)" for n in full] + [f"{n} (мин 5)" for n in full]
    names += ["позиция в эфире", "минут от начала", "минут до конца"]
    return names, nac, m


def group(n):
    for k, g in (("e5-", "семантика e5"), ("текст:", "метки текста"),
                 ("кадр:", "теги кадра"), ("кадр-вектор", "векторы кадра"),
                 ("звук:", "теги звука"), ("голос", "голос"),
                 ("Qwen:", "языковая модель"), ("застой", "видео: застой"),
                 ("яркость", "видео: простое"), ("движение", "видео: простое"),
                 ("смена плана", "видео: простое"),
                 ("позиция", "позиция"), ("минут", "позиция"),
                 ("слово:", "слова")):
        if n.startswith(k):
            return g
    return "акустика"


def main():
    bl = sys.argv[1] if len(sys.argv) > 1 else "ac+pca+tag+vid+vemb+xv+clap"
    blocks = set(bl.split("+"))
    X, D = get_data(10.0, blocks)
    slots = sorted(X)
    names, nac, m = all_names(blocks)
    if len(names) != X[slots[0]].shape[1]:
        print(f"ВНИМАНИЕ: имён {len(names)}, столбцов {X[slots[0]].shape[1]}")
        return
    imp = np.zeros(len(names))
    for sd in range(3):
        mdl = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05,
                                l2_leaf_reg=20, verbose=0, random_seed=sd,
                                thread_count=5, rsm=0.3, subsample=0.8,
                                bootstrap_type="Bernoulli")
        mdl.fit(np.vstack([X[s] for s in slots]),
                np.concatenate([D[s][0] for s in slots]),
                sample_weight=np.concatenate([D[s][1] for s in slots]))
        imp += mdl.get_feature_importance()
    imp /= 3

    g = collections.defaultdict(float)
    for n, v in zip(names, imp):
        g[group(n)] += v
    print("=== вклад по группам, % ===")
    for k, v in sorted(g.items(), key=lambda z: -z[1]):
        print(f"  {k:>18}  {v:>5.1f}")
    print("\n=== тридцать самых важных признаков ===")
    for i in np.argsort(-imp)[:30]:
        print(f"  {imp[i]:>5.2f}  {names[i]}")


if __name__ == "__main__":
    main()
