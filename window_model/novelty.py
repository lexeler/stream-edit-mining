#!/usr/bin/env python3
"""
Новизна: было ли такое раньше в этом же стриме.

Все прежние признаки описывают кусок сам по себе. Но монтажёр выбрасывает
не только скучное, а и ПОВТОРНОЕ: если про это уже говорили или мы это уже
видели, второй раз он не оставит. Плюс важно, сколько длится одно и то же —
«идут молча уже восемь минут» это почти наверняка под нож.

Считаем по уже готовым векторам (текст e5 и кадр CLIP), поэтому дёшево:

  похожесть на последнюю минуту / пять минут / полчаса
  наибольшая похожесть на что-либо РАНЬШЕ в этом стриме (повтор)
  сколько подряд окно похоже на предыдущее (длительность застоя)
"""
import os
import numpy as np

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
SLOTS = ["1", "2", "3", "4", "5", "6", "7"]
W = 10.0                       # сетка окон, с


def to_windows(E, times, n):
    """усредняем векторы по окнам сетки W"""
    idx = np.clip((times / W).astype(int), 0, n - 1)
    out = np.zeros((n, E.shape[1]), np.float32)
    cnt = np.zeros(n)
    np.add.at(out, idx, E)
    np.add.at(cnt, idx, 1)
    good = cnt > 0
    out[good] /= cnt[good, None]
    # пустые окна заполняем предыдущим
    last = None
    for i in range(n):
        if good[i]:
            last = out[i]
        elif last is not None:
            out[i] = last
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)


def feats(V):
    n = len(V)
    S = V @ V.T                                   # похожесть всех со всеми
    out = []
    for k in (6, 30, 180):                        # 1 мин, 5 мин, 30 мин
        v = np.zeros(n, np.float32)
        for i in range(n):
            lo = max(0, i - k)
            if i > lo:
                v[i] = S[i, lo:i].mean()
        out.append(v)
    # наибольшая похожесть на что-то раньше, но не на соседей
    mx = np.zeros(n, np.float32)
    for i in range(n):
        j = i - 30
        if j > 0:
            mx[i] = S[i, :j].max()
    out.append(mx)
    out.append(1 - mx)                            # новизна
    # длительность застоя: сколько подряд окно похоже на предыдущее
    prev = np.concatenate([[0.0], np.diag(S, -1)])
    thr = np.quantile(prev[1:], 0.6)
    run = np.zeros(n, np.float32)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if prev[i] > thr else 0
    out.append(np.minimum(run, 60))
    return np.stack(out, 1)


def main():
    sem = np.load(f"{DATA}/sem_raw.npy", allow_pickle=True).item()
    out = {}
    for s in SLOTS:
        d = sem[s]
        grid = d["grid"]
        n = int(np.ceil((grid[-1] + 2.0) / W))
        A = to_windows(d["emb"]["far"], grid, n)
        F = [feats(A)]
        try:
            e = np.load(f"{DATA}/clip_s{s}.npz")["emb"].astype(np.float32)
            t = np.arange(len(e)) / 2.0
            B = to_windows(e, t, n)
            F.append(feats(B))
        except Exception as ex:
            print(f"слот {s}: без кадров ({ex})")
        try:
            # новизна звуковой обстановки: по тегам CLAP
            a = np.load(f"{DATA}/clap_s{s}.npz")
            e = a["clap"].astype(np.float32)
            t = np.arange(len(e)) * float(a["clap_hop"])
            C = to_windows(e, t, n)
            F.append(feats(C))
        except Exception as ex:
            print(f"слот {s}: без звуковых тегов ({ex})")
        out[s] = np.hstack(F).astype(np.float32)
        print(f"слот {s}: окон {n}, признаков {out[s].shape[1]}", flush=True)
    np.save(f"{DATA}/novelty2.npy", out, allow_pickle=True)
    print("сохранено")


if __name__ == "__main__":
    main()
