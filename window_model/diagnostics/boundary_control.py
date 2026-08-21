#!/usr/bin/env python3
"""
КОНТРОЛЬНЫЙ ОПЫТ к boundary.py.

boundary.py дал AUC 0.922 и "в 9 раз точнее случайного", но контроля не было.
Подозрение: почти все границы монтажа стоят рядом с длинной паузой, и модель
просто находит паузы. Это было бы бесполезно — вырезать тишину мы и так умеем.

Проверяем три вещи:
  1. что даёт тупая эвристика "рядом длинная пауза" — без всякого обучения;
  2. что даёт лучший одиночный признак;
  3. и главное: разбивка по типу границы.
     речь|речь  — обе стороны слова, это настоящее содержательное решение;
     остальные  — хотя бы с одной стороны тишина, это техническая резка пауз.
"""
import os
import json, collections, sys
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SKIP = {"slot", "i", "type", "start", "end", "word",
        "used", "used_half", "used_frac", "cut_rel", "cut_side"}
W = 8


def build(rows, sem, base):
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    S = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idx in by.items():
        d = sem[s]
        S[idx] = np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]])
    B = np.array([[r[k] for k in base] for r in rows], dtype=np.float32)
    return B, S, by


def edges(B, S, y, isw, dur, by, slot, use_sem=True):
    """пары соседей внутри слота -> признаки перепада, метка, тип границы"""
    idx = np.array(by[slot])
    X = np.hstack([B, S]) if use_sem else B
    n = len(idx)
    F, Y, T, P = [], [], [], []
    for k in range(n - 1):
        a, b = idx[k], idx[k + 1]
        lo = idx[max(0, k - W + 1):k + 1]
        hi = idx[k + 1:k + 1 + W]
        L, R = X[lo].mean(0), X[hi].mean(0)
        F.append(np.concatenate([R - L, np.abs(R - L), X[b] - X[a]]))
        Y.append(int((y[a] > 0.5) != (y[b] > 0.5)))
        T.append(int(isw[a] and isw[b]))            # 1 = граница речь|речь
        # тупая эвристика: длительность прилегающей тишины
        P.append(max(0.0 if isw[a] else dur[a], 0.0 if isw[b] else dur[b]))
    return (np.array(F, np.float32), np.array(Y, np.int8),
            np.array(T, np.int8), np.array(P, np.float32))


def report(name, y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return (roc_auc_score(y, p), average_precision_score(y, p), y.mean(), len(y))


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()
    base = [k for k in rows[0] if k not in SKIP]
    B, S, by = build(rows, sem, base)
    y = np.array([r["used_frac"] for r in rows], np.float32)
    isw = np.array([r["is_word"] for r in rows], np.int8)
    dur = np.array([r["dur"] for r in rows], np.float32)
    slots = sorted(by)

    E, Es, L, T, P = {}, {}, {}, {}, {}
    for s in slots:
        E[s], L[s], T[s], P[s] = edges(B, S, y, isw, dur, by, s, use_sem=False)
        Es[s], _, _, _ = edges(B, S, y, isw, dur, by, s, use_sem=True)

    print("=== из чего вообще состоят границы монтажа ===")
    print(f"{'слот':>5}{'границ':>9}{'склеек':>8}{'доля':>8}"
          f"{'речь|речь':>11}{'их доля':>9}")
    print("-" * 50)
    for s in slots:
        cut = L[s].astype(bool)
        sp = (T[s] == 1) & cut
        print(f"{s:>5}{len(L[s]):>9}{cut.sum():>8}{cut.mean()*100:>7.2f}%"
              f"{sp.sum():>11}{sp.sum()/max(cut.sum(),1)*100:>8.1f}%")

    print("\n=== контроль: тупая эвристика 'рядом длинная пауза' ===")
    print("без всякого обучения, score = длительность прилегающей тишины")
    print(f"{'слот':>5}{'AUC':>8}{'AP':>8}{'случ.AP':>10}{'раз':>7}")
    print("-" * 38)
    hb = []
    for s in slots:
        r = report("", L[s], P[s])
        hb.append(r[0])
        print(f"{s:>5}{r[0]:>8.3f}{r[1]:>8.3f}{r[2]:>10.4f}{r[1]/r[2]:>6.1f}x")
    print(f"{'сред':>5}{np.mean(hb):>8.3f}")

    for use_sem, tag, EE in ((False, "только 28 признаков", E),
                             (True, "28 + семантика", Es)):
        print(f"\n=== модель, {tag} ===")
        print(f"{'слот':>5}{'AUC все':>10}{'AUC речь|речь':>15}"
              f"{'AP речь':>10}{'случ':>8}{'раз':>7}")
        print("-" * 55)
        a_all, a_sp, l_sp = [], [], []
        for held in slots:
            tr = [s for s in slots if s != held]
            Xtr = np.vstack([EE[s] for s in tr])
            ytr = np.concatenate([L[s] for s in tr])
            m = CatBoostClassifier(iterations=1200, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=8, verbose=0, random_seed=0,
                                   auto_class_weights="Balanced",
                                   thread_count=10)
            m.fit(Xtr, ytr)
            p = m.predict_proba(EE[held])[:, 1]
            ra = report("", L[held], p)
            rs = report("", L[held], p, mask=(T[held] == 1))
            a_all.append(ra[0]); a_sp.append(rs[0]); l_sp.append(rs[1] / rs[2])
            print(f"{held:>5}{ra[0]:>10.3f}{rs[0]:>15.3f}"
                  f"{rs[1]:>10.3f}{rs[2]:>8.4f}{rs[1]/rs[2]:>6.1f}x")
        print("-" * 55)
        print(f"{'сред':>5}{np.mean(a_all):>10.3f}{np.mean(a_sp):>15.3f}"
              f"{'':>10}{'':>8}{np.mean(l_sp):>6.1f}x")

    print("\nЧитать так: если 'AUC все' у модели близок к эвристике —")
    print("модель ничего не добавила, всё держится на паузах.")
    print("'AUC речь|речь' — настоящий показатель: умеет ли она резать речь.")


if __name__ == "__main__":
    main()
