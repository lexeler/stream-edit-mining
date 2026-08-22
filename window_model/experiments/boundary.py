#!/usr/bin/env python3
"""
Смена постановки: ищем не метки элементов, а МЕСТА СКЛЕЕК.

Задача в нынешнем виде вырождена: 97.7% ответов — "как было у предыдущего".
Настоящих решений монтажёра всего около 800 переходов на слот. Учим модель
отвечать на другой вопрос: проходит ли граница монтажа между этими двумя
соседними элементами.

Признаки — не свойства элемента, а ПЕРЕПАД через границу: насколько
отличается то, что слева, от того, что справа.
"""
import os
import json, collections
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}
W = 8          # сколько элементов с каждой стороны усредняем


def build(rows, sem, base):
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idx in by.items():
        d = sem[s]
        S[idx] = np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]])
    B = np.array([[r[k] for k in base] for r in rows], dtype=np.float32)
    return np.hstack([B, S]), by


def edges(X, y, by, slot):
    """пары соседних элементов внутри слота -> признаки перепада и метка"""
    idx = np.array(by[slot])
    F, Y = [], []
    n = len(idx)
    for k in range(n - 1):
        a, b = idx[k], idx[k + 1]
        lo = idx[max(0, k - W + 1):k + 1]
        hi = idx[k + 1:k + 1 + W]
        L, R = X[lo].mean(0), X[hi].mean(0)
        F.append(np.concatenate([R - L, np.abs(R - L), X[b] - X[a]]))
        # граница монтажа: слева и справа разные решения
        Y.append(int((y[a] > 0.5) != (y[b] > 0.5)))
    return np.array(F, dtype=np.float32), np.array(Y, dtype=np.int8)


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]
    X, by = build(rows, sem, base)
    y = np.array([r["used_frac"] for r in rows], np.float32)
    slots = sorted(by)

    E, L = {}, {}
    for s in slots:
        E[s], L[s] = edges(X, y, by, s)
        print(f"слот {s}: {len(L[s])} границ, из них склеек {L[s].sum()} "
              f"({L[s].mean()*100:.2f}%)", flush=True)

    print(f"\nпризнаков на границу: {E[slots[0]].shape[1]}")
    print("\nпроверка: находит ли модель места склеек")
    print(f"{'слот':>5}{'AUC':>8}{'AP':>8}{'случайно AP':>13}{'во сколько раз':>16}")
    print("-" * 52)
    aucs, lifts = [], []
    for held in slots:
        tr = [s for s in slots if s != held]
        Xtr = np.vstack([E[s] for s in tr])
        ytr = np.concatenate([L[s] for s in tr])
        m = CatBoostClassifier(iterations=1200, depth=6, learning_rate=0.05,
                               l2_leaf_reg=8, verbose=0, random_seed=0,
                               auto_class_weights="Balanced")
        m.fit(Xtr, ytr)
        p = m.predict_proba(E[held])[:, 1]
        auc = roc_auc_score(L[held], p)
        ap = average_precision_score(L[held], p)
        rnd = L[held].mean()
        aucs.append(auc); lifts.append(ap / rnd)
        print(f"{held:>5}{auc:>8.3f}{ap:>8.3f}{rnd:>13.4f}{ap/rnd:>15.1f}x")
    print("-" * 52)
    print(f"{'сред':>5}{np.mean(aucs):>8.3f}{'':>8}{'':>13}{np.mean(lifts):>15.1f}x")
    print("\nAUC 0.5 = монетка. Во сколько раз — насколько точнее случайного тыка.")


if __name__ == "__main__":
    main()
