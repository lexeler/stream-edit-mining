#!/usr/bin/env python3
"""
Дообучение текстового кодировщика прямо на задачу.

Замороженные векторы e5 дают мало: сжатие до 96 чисел прибавило 0,02 к F1,
смысловые метки столько же. Гипотеза: дело не в тексте, а в том, что e5
кодирует «о чём речь», а не «что важно этому монтажёру». Проверяем прямо:
учим маленький кодировщик предсказывать долю оставленного по тексту окна.

Опасность понятна и её надо держать в голове: обучающих стримов пять,
модель легко выучит имена и места конкретных эфиров вместо общего правила.
Поэтому кодировщик маленький, эпох мало, и проверка по отложенному слоту.
"""
import json, collections, sys, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "intfloat/multilingual-e5-small"
WSEC, CTX = 10.0, 30.0
MAXLEN = 192
EPOCHS = 3
BS = 32
LR = 2e-5


class WinDS(Dataset):
    def __init__(self, txt, y, tok):
        self.t, self.y, self.tok = txt, y, tok

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.y[i]


def collate(b, tok):
    txt = [x[0] for x in b]
    y = torch.tensor([x[1] for x in b], dtype=torch.float32)
    e = tok(txt, return_tensors="pt", padding=True, truncation=True,
            max_length=MAXLEN)
    return e, y


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.drop = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **b):
        o = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        v = (o * m).sum(1) / m.sum(1)
        return self.fc(self.drop(v)).squeeze(-1)


def build(slot, rows):
    dur = np.array([r["dur"] for r in rows], np.float32)
    st = np.array([r["start"] for r in rows], np.float32)
    frac = np.array([r["used_frac"] for r in rows], np.float32)
    T = float(st[-1] + dur[-1])
    nw = int(np.ceil(T / WSEC))
    wid = np.clip(((st + dur / 2) / WSEC).astype(int), 0, nw - 1)
    words = [(r["start"], r["word"]) for r in rows if r["type"] == "word"]
    ws = np.array([w[0] for w in words]) if words else np.zeros(1)
    Y = np.zeros(nw, np.float32); WT = np.zeros(nw, np.float32)
    for w in range(nw):
        m = wid == w
        if m.any():
            Y[w] = (dur[m] * frac[m]).sum() / dur[m].sum()
            WT[w] = dur[m].sum()
    txt = []
    for w in range(nw):
        c = (w + 0.5) * WSEC
        lo = int(np.searchsorted(ws, c - CTX)); hi = int(np.searchsorted(ws, c + CTX))
        txt.append("query: " + " ".join(x[1] for x in words[lo:hi])[:900])
    return txt, Y, WT


def main():
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    slots = sorted(by)
    D = {s: build(s, by[s]) for s in slots}
    print("окон: " + ", ".join(f"{s}:{len(D[s][0])}" for s in slots), flush=True)

    out = {}
    aucs = []
    for held in slots:
        tr = [s for s in slots if s != held]
        Xt = sum([D[s][0] for s in tr], []); yt = np.concatenate([D[s][1] for s in tr])
        ds = WinDS(Xt, yt, tok)
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate(b, tok))
        mdl = Head().to(dev)
        opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=EPOCHS * len(dl), pct_start=0.1)
        mdl.train()
        for ep in range(EPOCHS):
            tl = 0.0
            for e, y in dl:
                e = {k: v.to(dev) for k, v in e.items()}
                p = mdl(**e)
                loss = torch.nn.functional.mse_loss(p, y.to(dev))
                loss.backward(); opt.step(); sch.step(); opt.zero_grad()
                tl += float(loss)
            print(f"  слот {held} эпоха {ep}: потери {tl/len(dl):.4f}", flush=True)
        mdl.eval()
        te = DataLoader(WinDS(D[held][0], D[held][1], tok), batch_size=64,
                        collate_fn=lambda b: collate(b, tok))
        ps = []
        with torch.no_grad():
            for e, _ in te:
                ps.append(mdl(**{k: v.to(dev) for k, v in e.items()}).cpu().numpy())
        p = np.concatenate(ps)
        out[held] = p
        a = roc_auc_score(D[held][1] > 0.5, p)
        aucs.append(a)
        print(f"слот {held}: AUC по одному тексту {a:.3f}", flush=True)
        del mdl
        torch.cuda.empty_cache()
    print(f"\nсредний AUC только по тексту: {np.mean(aucs):.3f}")
    np.save(f"{DATA}/textft.npy", {s: out[s] for s in out}, allow_pickle=True)
    print("предсказания сохранены, дальше их можно подать как признак")


if __name__ == "__main__":
    main()
