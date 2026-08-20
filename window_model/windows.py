#!/usr/bin/env python3
"""
Макро-модель, рабочая версия. Признаки кэшируются, опыты идут за секунды.

Блоки признаков:
  ac   акустика и слова из ленты элементов
  pca  сжатая семантика текста e5 (48+48)
  tag  смысловые метки текста, 22 вопроса × 2 масштаба
  vid  CLIP: 24 тега кадра, визуальный застой, движение и яркость
  aud  CLAP: 16 тегов звука; изменчивость голоса по x-вектору

Потери:
  rmse   регрессия на долю
  log    классификация оставлено/нет
  rank   ранжирование внутри стрима (задача и есть «отобрать лучшие N %»)

Решение: доля выпуска выбирается моделью самостоятельно, затем сглаживание
по Витерби. Подсказок о длительности нет.

  windows.py --w 10 --blocks ac+pca+tag+vid+aud --loss rank
"""
import json, collections, sys, os, argparse, hashlib
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
CACHE = os.path.join(os.environ.get("SEM_WORK", "work/window"), "cache")
AGG_MEAN = ["vol", "vol_std", "zcr", "centroid", "flatness", "voice",
            "dens5", "dens15", "dens60", "vrel5", "vrel15", "vrel60",
            "pause_dens", "d_dens", "d_vol"]
AGG_WORD = ["wlen", "is_filler", "has_mat", "ends_sent", "is_question",
            "is_caps", "is_repeat"]


# ------------------------------------------------------- сборка признаков
def ext_features(slot, rr, blocks):
    """дополнительные признаки, приведённые к элементам ленты"""
    mid = np.array([(r["start"] + r["end"]) / 2 for r in rr], np.float32)
    out = []
    if "pca" in blocks:
        d = np.load(f"{DATA}/sem_pca.npy", allow_pickle=True).item()[slot]
        out.append(np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]]))
    if "tag" in blocks:
        d = np.load(f"{DATA}/semtags.npy", allow_pickle=True).item()[slot]
        out.append(np.hstack([d["near"][d["idx"]], d["far"][d["idx"]]]))
    if "vid" in blocks:
        p = f"{DATA}/clip_s{slot}.npz"
        if os.path.exists(p):
            v = np.load(p)
            fi = np.clip((mid * 2).astype(int), 0, int(v["n"]) - 1)
            cols = [v["tags"][fi].astype(np.float32)]
            for k in ("s30", "s120", "s300"):
                cols.append(v[k][fi][:, None].astype(np.float32))
            w = np.load(os.path.join(os.environ.get("SEM_WORK", "work/window"), f"vid/vfeat_s{slot}.npz"))
            gi = np.clip((mid * 2).astype(int), 0, len(w["bright"]) - 1)
            for k in ("bright", "motion", "scene"):
                cols.append(w[k][gi][:, None].astype(np.float32))
            out.append(np.hstack(cols))
    if "vemb" in blocks:
        p = f"{DATA}/clip_pca.npy"
        if os.path.exists(p):
            e = np.load(p, allow_pickle=True).item()[slot]
            fi = np.clip((mid * 2).astype(int), 0, len(e) - 1)
            out.append(e[fi])
    if "vtag2" in blocks:
        p = f"{DATA}/cliptag2.npy"
        if os.path.exists(p):
            e = np.load(p, allow_pickle=True).item()[slot]
            fi = np.clip((mid * 2).astype(int), 0, len(e) - 1)
            out.append(e[fi])
    if "clap" in blocks:
        p = f"{DATA}/clap_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p)
            ci = np.clip((mid / float(a["clap_hop"])).astype(int), 0, len(a["clap"]) - 1)
            out.append(a["clap"][ci].astype(np.float32))
    if "xv" in blocks:
        # Изменчивость голоса (расстояние между соседними отпечатками) не
        # работает: контроль xvec_check.py показал, что на уличном звуке
        # отпечаток почти не различает говорящих (соседние 0,75 против
        # случайных 0,70). Зато расстояние до «хозяина эфира» разделяет
        # одинаково на всех шести слотах: оставленное — это чистая речь
        # ведущего, выброшенное дальше от его среднего голоса.
        p = f"{DATA}/xvec_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p)
            h = float(a["xv_hop"])
            xv = a["xv"].astype(np.float32)
            t = np.arange(len(xv)) * h
            st = np.array([r["start"] for r in rr], np.float32)
            isw = np.array([r["is_word"] for r in rr], np.int8)
            j = np.clip(np.searchsorted(st, t) - 1, 0, len(rr) - 1)
            sp = isw[j] == 1
            host = xv[sp].mean(0) if sp.any() else xv.mean(0)
            host = host / (np.linalg.norm(host) + 1e-6)
            d = (1 - xv @ host).astype(np.float32)
            xi = np.clip((mid / h).astype(int), 0, len(d) - 1)
            cols = [d[xi][:, None], a["dxv"].astype(np.float32)[xi][:, None]]
            c = np.concatenate([[0.0], np.cumsum(d)])
            for sec in (15, 60, 300):
                k = max(int(sec / h), 1)
                lo = np.clip(xi - k, 0, len(d)); hi = np.clip(xi + k + 1, 0, len(d))
                cols.append(((c[hi] - c[lo]) / np.maximum(hi - lo, 1))[:, None])
            out.append(np.hstack(cols).astype(np.float32))
    if "surp" in blocks:
        p = f"{DATA}/surp_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            g = float(a["grid"]); F = a["feats"].astype(np.float32)
            si = np.clip((mid / g).astype(int), 0, len(F) - 1)
            out.append(F[si])
    if "icl" in blocks:
        p = f"{DATA}/icl2_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            h = float(a["hop"])
            ii = np.clip((mid / h).astype(int), 0, len(a["scores"]) - 1)
            out.append(a["scores"][ii].astype(np.float32))
    if "mom" in blocks:
        p = f"{DATA}/moment_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            h = float(a["hop"])
            mi = np.clip((mid / h).astype(int), 0, len(a["scores"]) - 1)
            out.append(a["scores"][mi].astype(np.float32))
    if "nov2" in blocks:
        p = f"{DATA}/novelty2.npy"
        if os.path.exists(p):
            e = np.load(p, allow_pickle=True).item()[slot]
            ni = np.clip((mid / 10.0).astype(int), 0, len(e) - 1)
            out.append(e[ni])
    if "pros" in blocks:
        p = f"{DATA}/pros_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            g = float(a["grid"]); F = a["feats"].astype(np.float32)
            pi = np.clip((mid / g).astype(int), 0, len(F) - 1)
            out.append(F[pi])
    if "nov" in blocks:
        p = f"{DATA}/novelty.npy"
        if os.path.exists(p):
            e = np.load(p, allow_pickle=True).item()[slot]
            ni = np.clip((mid / 10.0).astype(int), 0, len(e) - 1)
            out.append(e[ni])
    if "scene" in blocks:
        p = f"{DATA}/scene_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p, allow_pickle=True)
            sc = float(a["scene"])
            si = np.clip((mid / sc).astype(int), 0, len(a["scores"]) - 1)
            out.append(a["scores"][si].astype(np.float32))
    if "llm" in blocks:
        p = f"{DATA}/llm_s{slot}.npz"
        if os.path.exists(p):
            a = np.load(p)
            h = float(a["hop"])
            li = np.clip((mid / h).astype(int), 0, len(a["scores"]) - 1)
            out.append(a["scores"][li].astype(np.float32))
    return np.hstack(out) if out else np.zeros((len(rr), 0), np.float32)


def build_windows(rr, EXT, wsec):
    dur = np.array([r["dur"] for r in rr], np.float32)
    st = np.array([r["start"] for r in rr], np.float32)
    frac = np.array([r["used_frac"] for r in rr], np.float32)
    isw = np.array([r["is_word"] for r in rr], np.int8)
    T = st[-1] + dur[-1]
    nw = int(np.ceil(T / wsec))
    wid = np.clip((st + dur / 2) / wsec, 0, nw - 1).astype(np.int32)
    cols = {k: np.array([r[k] for r in rr], np.float32) for k in AGG_MEAN + AGG_WORD}
    order = np.argsort(wid, kind="stable")
    b = np.searchsorted(wid[order], np.arange(nw + 1))
    F, Y, WT, nf, nac = [], [], [], None, None
    for w in range(nw):
        m = order[b[w]:b[w + 1]]
        if len(m) == 0:
            F.append(None); Y.append(0.0); WT.append(0.0); continue
        d = dur[m]; tot = d.sum()
        f = [tot, (d * isw[m]).sum() / tot, float(isw[m].sum()),
             isw[m].sum() / tot * 60]
        pd = d[isw[m] == 0]
        f += [pd.sum() / tot, float(pd.max()) if len(pd) else 0.0,
              float((pd > 1).sum()), float((pd > 3).sum())]
        for k in AGG_MEAN:
            v = cols[k][m]
            f += [float((v * d).sum() / tot), float(v.std())]
        wm = m[isw[m] == 1]
        f += ([float(cols[k][wm].mean()) for k in AGG_WORD] if len(wm)
              else [0.0] * len(AGG_WORD))
        nac = len(f)
        if EXT.shape[1]:
            f += list((EXT[m] * d[:, None]).sum(0) / tot)
        F.append(np.array(f, np.float32)); nf = len(f)
        Y.append(float((d * frac[m]).sum() / tot)); WT.append(float(tot))
    F = [x if x is not None else np.zeros(nf, np.float32) for x in F]
    return (np.array(F, np.float32), np.array(Y, np.float32),
            np.array(WT, np.float32), nac)


def context(F, wsec, nac, qn=False):
    n, m = F.shape
    A = F[:, :nac]
    if qn:
        # ранговая нормировка внутри стрима: у разных эфиров разная форма
        # распределений, а не только среднее и разброс. После перевода
        # в ранги вход модели одинаков для всех стримов
        R = np.argsort(np.argsort(F, 0), 0).astype(np.float32) / max(n - 1, 1)
        parts = [F, R]
    else:
        parts = [F, (F - np.median(F, 0)) / (F.std(0) + 1e-6)]
    for lag in (1, 2, 4, 8, 16):
        parts.append(np.vstack([np.repeat(A[:1], lag, 0), A[:-lag]]) - A)
        parts.append(np.vstack([A[lag:], np.repeat(A[-1:], lag, 0)]) - A)
    for k in (max(int(60 / wsec), 1), max(int(300 / wsec), 1)):
        c = np.cumsum(np.vstack([np.zeros((1, m)), F]), 0)
        lo = np.clip(np.arange(n) - k, 0, n); hi = np.clip(np.arange(n) + k + 1, 0, n)
        parts.append(F - (c[hi] - c[lo]) / (hi - lo)[:, None])
    t = np.arange(n) * wsec
    parts.append(np.stack([np.arange(n) / n, np.minimum(t / 60, 30),
                           np.minimum((n - np.arange(n)) * wsec / 60, 30)], 1))
    return np.hstack(parts).astype(np.float32)


def get_data(wsec, blocks, split="train"):
    key = hashlib.md5(f"{split}|{wsec}|{'+'.join(sorted(blocks))}".encode()).hexdigest()[:12]
    p = f"{CACHE}/w_{key}.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"].item(), z["D"].item()
    os.makedirs(CACHE, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{DATA}/{split}.jsonl")]
    if split == "test":
        ans = [json.loads(l) for l in open(f"{DATA}/test_answers.jsonl")]
        for r, a in zip(rows, ans):
            r["used_frac"] = a["used_frac"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    X, D = {}, {}
    for s in sorted(by):
        E = ext_features(s, by[s], blocks)
        F, Y, WT, nac = build_windows(by[s], E, wsec)
        X[s] = context(F, wsec, nac, qn="qn" in blocks); D[s] = (Y, WT)
    np.savez(p, X=X, D=D)
    return X, D


# ------------------------------------------------------------- решения
def score(keep, Y, WT):
    ov = float((keep * Y * WT).sum()); pred = float((keep * WT).sum())
    true = float((Y * WT).sum())
    p = ov / pred if pred else 0.0
    r = ov / true if true else 0.0
    return (p, r, 2 * p * r / (p + r) if p + r else 0.0,
            ov / (pred + true - ov) if pred + true - ov else 0.0, pred / WT.sum())


def take_rate(p, WT, rate):
    o = np.argsort(-p)
    c = np.cumsum(WT[o]) / WT.sum()
    keep = np.zeros(len(p), bool)
    keep[o[c <= max(rate, 1e-6)]] = True
    if not keep.any():
        keep[o[0]] = True
    return keep


def viterbi(sc, WT, lam):
    """sc — выгода оставить окно (может быть отрицательной). Штраф lam за смену."""
    n = len(sc)
    dp0 = dp1 = 0.0
    bk = np.zeros((n, 2), np.int8)
    d0 = np.zeros(n); d1 = np.zeros(n)
    dp0, dp1 = 0.0, sc[0]
    d0[0], d1[0] = dp0, dp1
    for i in range(1, n):
        a0, b0 = d0[i - 1], d1[i - 1] - lam
        if b0 > a0:
            d0[i] = b0; bk[i, 0] = 1
        else:
            d0[i] = a0; bk[i, 0] = 0
        a1, b1 = d1[i - 1], d0[i - 1] - lam
        if b1 > a1:
            d1[i] = b1 + sc[i]; bk[i, 1] = 0
        else:
            d1[i] = a1 + sc[i]; bk[i, 1] = 1
    st = 1 if d1[-1] > d0[-1] else 0
    out = np.zeros(n, bool)
    for i in range(n - 1, -1, -1):
        out[i] = bool(st); st = bk[i, st]
    return out


def viterbi_rate(p, WT, lam, rate):
    lo, hi = -1.0, 1.0
    keep = None
    for _ in range(22):
        mid = (lo + hi) / 2
        keep = viterbi(WT * (p - mid), WT, lam * WT.mean())
        got = (keep * WT).sum() / WT.sum()
        if got > rate:
            lo = mid
        else:
            hi = mid
        if abs(got - rate) < 0.003:
            break
    return keep


def fit_predict(Xtr, ytr, wtr, gtr, Xte, loss, seed=0):
    kw = dict(iterations=1500, depth=6, learning_rate=0.04, l2_leaf_reg=12,
              verbose=0, random_seed=seed, thread_count=5)
    if loss == "log":
        m = CatBoostClassifier(**kw)
        m.fit(Xtr, (ytr > 0.5).astype(int), sample_weight=wtr)
        return m.predict_proba(Xte)[:, 1]
    if loss == "rank":
        m = CatBoostRanker(loss_function="YetiRank", **kw)
        o = np.argsort(gtr, kind="stable")
        m.fit(Pool(Xtr[o], ytr[o], group_id=gtr[o], weight=wtr[o]))
        return m.predict(Xte)
    m = CatBoostRegressor(loss_function="RMSE", **kw)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return np.clip(m.predict(Xte), 0, 1)


def rank01(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=10.0)
    ap.add_argument("--blocks", default="ac+pca")
    ap.add_argument("--loss", default="rmse")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    blocks = set(a.blocks.split("+"))
    X, D = get_data(a.w, blocks)
    slots = sorted(X)

    P = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        Xtr = np.vstack([X[s] for s in tr])
        ytr = np.concatenate([D[s][0] for s in tr])
        wtr = np.concatenate([D[s][1] for s in tr])
        gtr = np.concatenate([np.full(len(D[s][0]), int(s)) for s in tr])
        ps = [fit_predict(Xtr, ytr, wtr, gtr, X[held], a.loss, sd)
              for sd in range(a.seeds)]
        P[held] = np.mean(ps, 0)

    # предсказанная доля: для ранжирования шкалы нет, берём среднюю по обучению
    rates = {}
    for held in slots:
        tr = [s for s in slots if s != held]
        if a.loss == "rank":
            rates[held] = float(np.mean([(D[s][0] * D[s][1]).sum() / D[s][1].sum()
                                         for s in tr]))
        else:
            rates[held] = float((P[held] * D[held][1]).sum() / D[held][1].sum())

    res = collections.defaultdict(list)
    for held in slots:
        tr = [s for s in slots if s != held]
        best, bl = -1, 0
        for lam in (0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
            f = np.mean([score(viterbi_rate(rank01(P[s]), D[s][1], lam, rates[s]),
                               *D[s])[2] for s in tr])
            if f > best:
                best, bl = f, lam
        Y, WT = D[held]; p = P[held]
        res["своя доля"].append(score(take_rate(p, WT, rates[held]), Y, WT))
        res["своя доля + Витерби"].append(
            score(viterbi_rate(rank01(p), WT, bl, rates[held]), Y, WT))
        res["_auc"].append(roc_auc_score(Y > 0.5, p))
        res["_rate"].append((rates[held], float((Y * WT).sum() / WT.sum())))
        res["_lam"].append(bl)

    rt = np.array(res["_rate"])
    print(f"\n### окно {a.w:g} с | {'+'.join(sorted(blocks))} | потери {a.loss} "
          f"| признаков {X[slots[0]].shape[1]}")
    print(f"AUC {np.mean(res['_auc']):.3f} | доля {rt[:,0].mean():.1%} против "
          f"{rt[:,1].mean():.1%} | Витерби {res['_lam']}")
    print(f"{'решение':>22}{'точн':>8}{'полн':>8}{'F1':>8}{'IoU':>8}{'оставл':>9}")
    print("-" * 63)
    for k in ("своя доля", "своя доля + Витерби"):
        m = np.mean(res[k], 0)
        print(f"{k:>22}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>8.3f}{m[4]:>8.1%}")
    print(f"{'случайно':>22}{'':>8}{'':>8}{0.375:>8.3f}{0.234:>8.3f}")
    print("  по слотам (Витерби): " + "  ".join(
        f"{s}:{v[2]:.3f}" for s, v in zip(slots, res["своя доля + Витерби"])))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
