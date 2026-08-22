#!/usr/bin/env python3
"""
Последовательностная модель по ленте окон.

Бустинг видит контекст только через сдвиги, которые я задал руками
(±1, 2, 4, 8, 16 окон и скользящие средние). Сеть по ленте может собрать
контекст сама и уловить структуру сцены — а решения монтажёра как раз
сценами и идут (автокорреляция 0,62 на 10 с, 0,33 на минуте).

Опасность: обучающих стримов пять. Поэтому модель маленькая, с сильным
прореживанием, обучение на случайных кусках ленты, ранняя остановка
по ОТДЕЛЬНОМУ обучающему слоту — отложенный слот не трогаем вообще.
"""
import json, collections, os, sys, hashlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # модули родительской папки
import windows as M

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
CACHE = os.path.join(os.environ.get("SEM_WORK", "work/window"), "cache")
WSEC = 10.0
CROP = 192
HID = 96
EPOCHS = 40
LR = 2e-3


def base_features(blocks, wsec=WSEC, split="train"):
    key = hashlib.md5(f"base|{split}|{wsec}|{'+'.join(sorted(blocks))}"
                      .encode()).hexdigest()[:12]
    p = f"{CACHE}/b_{key}.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["F"].item(), z["D"].item()
    rows = [json.loads(l) for l in open(f"{M.DATA}/{split}.jsonl")]
    if split == "test":
        ans = [json.loads(l) for l in open(f"{M.DATA}/test_answers.jsonl")]
        for r, a in zip(rows, ans):
            r["used_frac"] = a["used_frac"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    F, D = {}, {}
    for s in sorted(by):
        E = M.ext_features(s, by[s], blocks)
        f, y, wt, nac = M.build_windows(by[s], E, wsec)
        n = len(f)
        pos = np.stack([np.arange(n) / n,
                        np.minimum(np.arange(n) * wsec / 60, 30) / 30,
                        np.minimum((n - np.arange(n)) * wsec / 60, 30) / 30], 1)
        f = (f - np.median(f, 0)) / (f.std(0) + 1e-6)      # внутри стрима
        F[s] = np.hstack([np.clip(f, -6, 6), pos]).astype(np.float32)
        D[s] = (y, wt)
    np.savez(p, F=F, D=D)
    return F, D


class Net(nn.Module):
    """свёртки с растущим охватом плюс двусторонняя рекуррентная память"""
    def __init__(self, nf):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, HID), nn.GELU(), nn.Dropout(0.3))
        self.conv = nn.ModuleList([
            nn.Conv1d(HID, HID, 5, padding=2 * d, dilation=d)
            for d in (1, 2, 4, 8, 16)])
        self.norm = nn.ModuleList([nn.GroupNorm(4, HID) for _ in range(5)])
        self.drop = nn.Dropout(0.2)
        self.gru = nn.GRU(HID, HID // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(HID, 1)

    def forward(self, x):
        h = self.inp(x).transpose(1, 2)
        for c, n in zip(self.conv, self.norm):
            h = h + self.drop(torch.nn.functional.gelu(n(c(h))))
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h).squeeze(-1)


def crops(F, Y, WT, slots, rng, bs=8):
    xs, ys, ws = [], [], []
    for _ in range(bs):
        s = slots[rng.integers(len(slots))]
        n = len(Y[s])
        if n <= CROP:
            a = 0; sl = slice(0, n)
            pad = CROP - n
        else:
            a = int(rng.integers(0, n - CROP)); sl = slice(a, a + CROP); pad = 0
        x = F[s][sl]; y = Y[s][sl]; w = WT[s][sl]
        if pad:
            x = np.vstack([x, np.zeros((pad, x.shape[1]), np.float32)])
            y = np.concatenate([y, np.zeros(pad, np.float32)])
            w = np.concatenate([w, np.zeros(pad, np.float32)])
        xs.append(x); ys.append(y); ws.append(w)
    return (torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)),
            torch.tensor(np.stack(ws)))


def train_predict(F, D, tr, val, tgt, dev, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Y = {s: D[s][0] for s in D}; WT = {s: D[s][1] for s in D}
    net = Net(F[tr[0]].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * 30)
    best, best_state = -1, None
    for ep in range(EPOCHS):
        net.train()
        for _ in range(30):
            x, y, w = crops(F, Y, WT, tr, rng)
            p = net(x.to(dev))
            loss = (((p - y.to(dev)) ** 2) * w.to(dev)).sum() / w.to(dev).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        net.eval()
        with torch.no_grad():
            pv = net(torch.tensor(F[val][None]).to(dev))[0].cpu().numpy()
        a = roc_auc_score(Y[val] > 0.5, pv)
        if a > best:
            best = a
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(F[tgt][None]).to(dev))[0].cpu().numpy(), best


def main():
    blocks = set((sys.argv[1] if len(sys.argv) > 1
                  else "ac+pca+tag+vid+vemb+xv+clap").split("+"))
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("SEQGPU") else "cpu"
    torch.set_num_threads(5)
    F, D = base_features(blocks)
    slots = sorted(F)
    print(f"признаков на окно {F[slots[0]].shape[1]}, устройство {dev}", flush=True)
    P, aucs = {}, []
    for held in slots:
        rest = [s for s in slots if s != held]
        ps = []
        for k, val in enumerate(rest[:3]):          # три разных слота для остановки
            tr = [s for s in rest if s != val]
            p, va = train_predict(F, D, tr, val, held, dev, seed=k)
            ps.append(p)
        P[held] = np.mean(ps, 0)
        a = roc_auc_score(D[held][0] > 0.5, P[held])
        aucs.append(a)
        print(f"слот {held}: AUC {a:.3f}", flush=True)
    print(f"\nсредний AUC последовательностной модели: {np.mean(aucs):.3f}")
    np.save(f"{DATA}/seqpred.npy", P, allow_pickle=True)


if __name__ == "__main__":
    main()
