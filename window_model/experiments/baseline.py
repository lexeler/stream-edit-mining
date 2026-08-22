#!/usr/bin/env python3
"""
Базовые уровни. Без них невозможно понять, научилась модель чему-то или нет.

Метрика по ВРЕМЕНИ, не по элементам: переходов всего 2.3%, поэтому модель
"повтори предыдущую метку" даёт 97.7% точности по элементам и при этом
бесполезна.

Проверка честная: обучаемся на пяти слотах, проверяемся на шестом,
и так шесть раз. Тестовый слот 2 не трогаем.
"""
import os
import json, sys, collections
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data/train.jsonl")
FEATS = None


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_xy(rows, feats):
    X = np.array([[r[k] for k in feats] for r in rows], dtype=np.float32)
    y = np.array([r["used_frac"] for r in rows], dtype=np.float32)
    d = np.array([r["end"] - r["start"] for r in rows], dtype=np.float32)
    return X, y, d


def time_metrics(pred, y, d):
    """pred и y — доли 0..1, d — длительности. Считаем по времени."""
    inter = np.minimum(pred, y) * d
    union = np.maximum(pred, y) * d
    iou = inter.sum() / max(union.sum(), 1e-9)
    keep_pred = (pred * d).sum()
    keep_true = (y * d).sum()
    prec = inter.sum() / max(keep_pred, 1e-9)
    rec = inter.sum() / max(keep_true, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1,
            "pred_min": keep_pred / 60, "true_min": keep_true / 60}


def report(name, pred, y, d):
    m = time_metrics(np.clip(pred, 0, 1), y, d)
    print(f"  {name:<34} IoU {m['iou']:.3f}  P {m['precision']:.3f}  "
          f"R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
          f"минут {m['pred_min']:6.1f} / {m['true_min']:6.1f}")
    return m


def main():
    rows = load(DATA)
    slots = sorted({r["slot"] for r in rows})
    global FEATS
    skip = {"slot", "i", "type", "start", "end", "word",
            "used", "used_half", "used_frac", "cut_rel", "cut_side"}
    FEATS = [k for k in rows[0] if k not in skip]
    print(f"строк {len(rows)}, слотов {len(slots)}, признаков {len(FEATS)}")
    print(f"признаки: {', '.join(FEATS)}\n")

    from catboost import CatBoostRegressor

    agg = collections.defaultdict(list)
    for held in slots:
        tr = [r for r in rows if r["slot"] != held]
        te = [r for r in rows if r["slot"] == held]
        Xtr, ytr, dtr = to_xy(tr, FEATS)
        Xte, yte, dte = to_xy(te, FEATS)
        print(f"--- проверка на слоте {held} ({len(te)} элементов) ---")

        agg["константа"].append(
            report("константа (средняя доля)", np.full(len(te), ytr.mean()), yte, dte))
        agg["всё берём"].append(
            report("всё берём", np.ones(len(te)), yte, dte))

        pd_ = Xte[:, FEATS.index("pause_dens")]
        thr = np.median(Xtr[:, FEATS.index("pause_dens")])
        agg["порог pause_dens"].append(
            report("порог по pause_dens", (pd_ <= thr).astype(float), yte, dte))

        m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05,
                              loss_function="RMSE", verbose=0,
                              random_seed=0, l2_leaf_reg=6)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        agg["catboost"].append(report("catboost на 28 признаках", p, yte, dte))
        print()

    print("=" * 72)
    print("СРЕДНЕЕ ПО ШЕСТИ ПРОВЕРКАМ")
    for k, v in agg.items():
        print(f"  {k:<24} IoU {np.mean([x['iou'] for x in v]):.3f}  "
              f"F1 {np.mean([x['f1'] for x in v]):.3f}")

    Xall, yall, _ = to_xy(rows, FEATS)
    m = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05,
                          verbose=0, random_seed=0, l2_leaf_reg=6)
    m.fit(Xall, yall)
    imp = sorted(zip(FEATS, m.get_feature_importance()), key=lambda x: -x[1])
    print("\nвклад признаков (топ-12):")
    for k, v in imp[:12]:
        print(f"  {k:<14} {v:6.2f}")


if __name__ == "__main__":
    main()
