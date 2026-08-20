#!/usr/bin/env python3
"""
Семантика: превращаем текст вокруг элемента в числа.

Два масштаба окна — ближний (что говорят прямо сейчас) и дальний (о чём вообще
сцена). Считаем не для каждого элемента, а через каждые 2 секунды стрима:
у соседних слов окно почти одинаковое, считать 215 тысяч раз незачем.

Сжатие обязательно. Два вектора по 1024 дают 2048 признаков, а настоящих
решений монтажёра около 2400 — модель их просто запомнит. Жмём до 48 каждый.
Матрицу сжатия настраиваем ТОЛЬКО на обучающих слотах.
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "intfloat/multilingual-e5-large"
STEP = 2.0                    # шаг сетки окон, с
SCALES = {"near": 7.5, "far": 30.0}
NPCA = 48
BATCH = 64


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def windows_for(rows, dur):
    """тексты окон на сетке через STEP секунд, для каждого масштаба"""
    words = [(r["start"], r["end"], r.get("word", ""))
             for r in rows if r["type"] == "word"]
    ws = np.array([w[0] for w in words])
    grid = np.arange(0, dur + STEP, STEP)
    out = {}
    for name, rad in SCALES.items():
        txts = []
        for t in grid:
            lo = np.searchsorted(ws, t - rad)
            hi = np.searchsorted(ws, t + rad)
            txts.append("passage: " + " ".join(w[2] for w in words[lo:hi])[:1800])
        out[name] = txts
    return grid, out


@torch.no_grad()
def embed(texts, tok, mdl, dev):
    res = []
    for i in range(0, len(texts), BATCH):
        b = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                truncation=True, max_length=384).to(dev)
        o = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        e = (o * m).sum(1) / m.sum(1)
        res.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if i and i % (BATCH * 50) == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return np.vstack(res).astype(np.float32)


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval().to(dev).half()
    print("модель загружена", flush=True)

    store = {}
    for split in ("train", "test"):
        rows = load(f"{DATA}/{split}.jsonl")
        for slot in sorted({r["slot"] for r in rows}):
            sr = [r for r in rows if r["slot"] == slot]
            dur = max(r["end"] for r in sr)
            grid, tx = windows_for(sr, dur)
            print(f"слот {slot}: {len(sr)} элементов, {len(grid)} окон", flush=True)
            embs = {}
            for name, txts in tx.items():
                embs[name] = embed(txts, tok, mdl, dev)
                print(f"  {name}: {embs[name].shape}", flush=True)
            store[slot] = {"grid": grid, "emb": embs, "split": split,
                           "idx": np.searchsorted(grid, [
                               (r["start"] + r["end"]) / 2 for r in sr]).clip(0, len(grid) - 1)}
    np.save(f"{DATA}/sem_raw.npy", store, allow_pickle=True)
    print("сырые векторы сохранены", flush=True)

    # сжатие: настраиваем только на обучающих слотах
    from sklearn.decomposition import PCA
    out = {}
    for name in SCALES:
        tr = np.vstack([store[s]["emb"][name] for s in store
                        if store[s]["split"] == "train"])
        p = PCA(n_components=NPCA, random_state=0).fit(tr)
        print(f"{name}: сжатие {tr.shape[1]} -> {NPCA}, "
              f"сохранено {p.explained_variance_ratio_.sum()*100:.1f}% разброса")
        for s in store:
            out.setdefault(s, {})[name] = p.transform(store[s]["emb"][name]).astype(np.float32)
    np.save(f"{DATA}/sem_pca.npy",
            {s: {"idx": store[s]["idx"], **out[s]} for s in store}, allow_pickle=True)
    print("сжатые векторы сохранены")


if __name__ == "__main__":
    main()
