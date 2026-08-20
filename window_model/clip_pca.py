#!/usr/bin/env python3
"""
Сжатие векторов кадра CLIP. Сейчас из видео берутся только 24 готовых тега,
а сам вектор (768 чисел на кадр) лежит без дела. Жмём его до 32 компонент —
вдруг в картинке есть то, чего не покрыли вопросы.

Матрица сжатия настраивается ТОЛЬКО на обучающих слотах, как у текста.
"""
import os
import numpy as np
from sklearn.decomposition import PCA

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
TRAIN = ["1", "3", "4", "5", "6", "7"]
ALL = TRAIN + ["2"]
NPCA = 32


def main():
    rng = np.random.default_rng(0)
    parts = []
    for s in TRAIN:
        e = np.load(f"{DATA}/clip_s{s}.npz")["emb"]
        i = rng.choice(len(e), min(8000, len(e)), replace=False)
        parts.append(e[i].astype(np.float32))
    X = np.vstack(parts)
    p = PCA(n_components=NPCA, random_state=0).fit(X)
    print(f"сжатие 768 -> {NPCA}, сохранено "
          f"{p.explained_variance_ratio_.sum()*100:.1f} % разброса")
    out = {}
    for s in ALL:
        e = np.load(f"{DATA}/clip_s{s}.npz")["emb"].astype(np.float32)
        out[s] = p.transform(e).astype(np.float32)
        print(f"слот {s}: {out[s].shape}")
    np.save(f"{DATA}/clip_pca.npy", out, allow_pickle=True)
    print("сохранено")


if __name__ == "__main__":
    main()
