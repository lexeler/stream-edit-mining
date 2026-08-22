#!/usr/bin/env python3
"""
Признак "а что монтажёр сделал в похожих местах".

Для каждого элемента ищем в ДРУГИХ эфирах моменты с похожим смыслом и
смотрим, что монтажёр с ними сделал. Доля оставленных среди ближайших
становится новым признаком.

Приём для малых данных: модель начинает опираться не только на обобщение,
но и на прямую память. При двух тысячах решений это заметно помогает.

Утечки нет: соседи ищутся только среди ЧУЖИХ слотов. Иначе элемент нашёл
бы сам себя и признак стал бы ответом.
"""
import os
import json, collections
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
K = 20


def build(rows, sem):
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["slot"]].append(k)
    E = np.zeros((len(rows), 96), dtype=np.float32)
    for s, idxs in by.items():
        d = sem[s]
        E[idxs] = np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]])
    # нормируем, чтобы скалярное произведение = косинус
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E, by


def knn_feature(E, by, y, slots, target_slot, batch=2048):
    """для элементов target_slot ищем соседей среди всех прочих слотов"""
    src = np.concatenate([by[s] for s in slots if s != target_slot])
    tgt = np.array(by[target_slot])
    A = E[src]
    ysrc = y[src]
    out_mean = np.zeros(len(tgt), dtype=np.float32)
    out_sim = np.zeros(len(tgt), dtype=np.float32)
    for i in range(0, len(tgt), batch):
        Q = E[tgt[i:i + batch]]
        sim = Q @ A.T
        idx = np.argpartition(-sim, K, axis=1)[:, :K]
        rows = np.arange(len(Q))[:, None]
        s = sim[rows, idx]
        o = np.argsort(-s, axis=1)
        idx = idx[rows, o]
        s = s[rows, o]
        w = np.maximum(s, 0) + 1e-6
        out_mean[i:i + batch] = (ysrc[idx] * w).sum(1) / w.sum(1)
        out_sim[i:i + batch] = s.mean(1)
    return out_mean, out_sim


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    te_rows = [json.loads(l) for l in open(f"{DATA}/test.jsonl")]
    ans = [json.loads(l) for l in open(f"{DATA}/test_answers.jsonl")]
    sem = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()

    allrows = rows + te_rows
    E, by = build(allrows, sem)
    y = np.array([r["used_frac"] for r in rows] + [0.0] * len(te_rows), np.float32)
    train_slots = sorted({r["slot"] for r in rows})

    out = {}
    for s in train_slots:
        m, sim = knn_feature(E, by, y, train_slots, s)
        out[s] = {"knn_used": m, "knn_sim": sim}
        print(f"слот {s}: соседи из {len(train_slots)-1} чужих эфиров, "
              f"средняя близость {sim.mean():.3f}, "
              f"разброс признака {m.std():.3f}", flush=True)

    # для теста соседи ищутся среди ВСЕХ обучающих
    ts = te_rows[0]["slot"]
    src = np.concatenate([by[s] for s in train_slots])
    tgt = np.array(by[ts])
    A, ysrc = E[src], y[src]
    m = np.zeros(len(tgt), np.float32)
    sm = np.zeros(len(tgt), np.float32)
    for i in range(0, len(tgt), 2048):
        Q = E[tgt[i:i + 2048]]
        sim = Q @ A.T
        idx = np.argpartition(-sim, K, axis=1)[:, :K]
        r = np.arange(len(Q))[:, None]
        s = sim[r, idx]
        w = np.maximum(s, 0) + 1e-6
        m[i:i + 2048] = (ysrc[idx] * w).sum(1) / w.sum(1)
        sm[i:i + 2048] = s.mean(1)
    out[ts] = {"knn_used": m, "knn_sim": sm}
    print(f"слот {ts} (тест): средняя близость {sm.mean():.3f}")

    np.save(f"{DATA}/knn.npy", out, allow_pickle=True)
    print("сохранено knn.npy")

    # быстрая проверка полезности: связь признака с ответом на обучении
    for s in train_slots[:3]:
        idx = np.array(by[s])
        yy = y[idx]
        kk = out[s]["knn_used"]
        c = np.corrcoef(kk, yy)[0, 1]
        print(f"  слот {s}: корреляция признака с ответом {c:+.3f}")


if __name__ == "__main__":
    main()
