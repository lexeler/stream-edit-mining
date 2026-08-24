#!/usr/bin/env python3
"""
Полные признаки на блок-реплику: всё лучшее из прежнего конвейера
плюс типы от языковой модели.

Прежний конвейер считал признаки на сетке 10 секунд (потолок точности 0,827).
Здесь та же начинка ложится на реплики (потолок 0,871): границы реплик
совпадают с местами, где монтажёр реально режет — он ставит склейку в паузе.

Блоки берутся из data/blocks/s<N>.blocks.json, остальное — из $SEM_WORK/data,
где уже посчитаны e5, CLIP, CLAP, x-вектор, вопросы к языковой модели,
новизна и просодия.

  python3 feats_server.py --slot 7
"""
import json, os, re, math, argparse
from collections import Counter
import numpy as np

CUT = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
RULES = os.environ.get("SEM_BLOCKS", "work/blocks")
TYPES = list("AGRPDWTSFX")
FLAGS = list("!rn")


def grid_mean(vals, hop, blocks, offset=0.0):
    """среднее значение сеточного признака по каждому блоку"""
    n = len(blocks)
    m = vals.shape[1] if vals.ndim > 1 else 1
    V = vals.reshape(len(vals), m)
    out = np.zeros((n, m), np.float32)
    for k, b in enumerate(blocks):
        i0 = int((b["start"] - offset) / hop)
        i1 = int(math.ceil((b["end"] - offset) / hop))
        i0 = max(0, min(i0, len(V) - 1)); i1 = max(i0 + 1, min(i1, len(V)))
        out[k] = V[i0:i1].mean(0)
    return out


def roll(vals, mid, dur, sec):
    n = len(vals)
    c = np.concatenate([[0.0], np.cumsum(vals * dur)])
    cd = np.concatenate([[0.0], np.cumsum(dur)])
    lo = np.searchsorted(mid, mid - sec)
    hi = np.searchsorted(mid, mid + sec)
    d = cd[hi] - cd[lo]
    return np.where(d > 0, (c[hi] - c[lo]) / np.maximum(d, 1e-9), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True)
    ap.add_argument("--marks", default="m1")
    a = ap.parse_args()
    s = a.slot
    B = json.load(open(f"{RULES}/blocks/s{s}.blocks.json"))["blocks"]
    n = len(B)
    dur = np.array([b["dur"] for b in B], np.float32)
    start = np.array([b["start"] for b in B], np.float32)
    mid = start + dur / 2
    T = float(start[-1] + dur[-1])
    is_sp = np.array([b["kind"] == "speech" for b in B], np.float32)
    text = [b["text"] for b in B]
    nchar = np.array([len(t) for t in text], np.float32)
    nword = np.array([len(t.split()) for t in text], np.float32)

    cols, names = [], []

    def add(M, nm):
        M = np.asarray(M, np.float32)
        if M.ndim == 1:
            M = M[:, None]
        cols.append(M)
        names.extend([f"{nm}{i}" for i in range(M.shape[1])] if M.shape[1] > 1
                     else [nm])

    # ---------- форма речи: главный предсказатель на границах
    dens_char = nchar / np.maximum(dur, 0.3)
    dens_word = nword / np.maximum(dur, 0.3)
    pause_before = np.zeros(n, np.float32); pause_after = np.zeros(n, np.float32)
    for i in range(1, n):
        pause_before[i] = dur[i - 1] if B[i - 1]["kind"] == "pause" else 0.0
    for i in range(n - 1):
        pause_after[i] = dur[i + 1] if B[i + 1]["kind"] == "pause" else 0.0
    add(dur, "dur"); add(nchar, "nchar"); add(nword, "nword")
    add(dens_char, "dens_char"); add(dens_word, "dens_word")
    add(nchar / np.maximum(nword, 1), "wlen"); add(is_sp, "is_speech")
    add(pause_before, "pause_before"); add(pause_after, "pause_after")
    add(mid / T, "pos"); add(np.minimum(mid / 60, 40), "min_from_start")
    add(np.minimum((T - mid) / 60, 40), "min_to_end")

    # ---------- новизна слов внутри стрима
    seen = Counter(); nov = np.zeros(n, np.float32)
    for i in range(n):
        w = set(re.findall(r"\w{4,}", text[i].lower()))
        if w:
            nov[i] = sum(1 for x in w if seen[x] == 0) / len(w)
        for x in w:
            seen[x] += 1
    add(nov, "novelty_words")

    # ---------- готовые признаки прежнего конвейера
    def try_load(fn):
        try:
            return fn()
        except Exception as e:
            print(f"  пропуск: {e}")
            return None

    # семантика e5 (сжатая) и смысловые метки — по словам, сетка из sem_raw
    d = try_load(lambda: np.load(f"{CUT}/sem_pca.npy", allow_pickle=True).item()[s])
    if d is not None:
        # near/far лежат на своей сетке grid (по одной точке на окно текста),
        # а поле idx — отображение элемента ленты в номер точки. Нам нужна
        # сама сетка: берём точки, попавшие в блок
        g = np.load(f"{CUT}/sem_raw.npy", allow_pickle=True).item()[s]["grid"]
        for key in ("near", "far"):
            V = d[key]
            M = np.zeros((n, V.shape[1]), np.float32)
            for k, b in enumerate(B):
                m = (g >= b["start"]) & (g < b["end"])
                M[k] = V[m].mean(0) if m.any() else V[
                    min(int(np.searchsorted(g, b["start"])), len(V) - 1)]
            add(M, f"e5_{key}_")
    d = try_load(lambda: np.load(f"{CUT}/semtags.npy", allow_pickle=True).item()[s])
    if d is not None:
        g = np.load(f"{CUT}/sem_raw.npy", allow_pickle=True).item()[s]["grid"]
        for key in ("near", "far"):
            V = d[key]
            M = np.zeros((n, V.shape[1]), np.float32)
            for k, b in enumerate(B):
                m = (g >= b["start"]) & (g < b["end"])
                M[k] = V[m].mean(0) if m.any() else V[
                    min(int(np.searchsorted(g, b["start"])), len(V) - 1)]
            add(M, f"tag_{key}_")

    z = try_load(lambda: np.load(f"{CUT}/clip_s{s}.npz"))
    if z is not None:
        add(grid_mean(z["tags"].astype(np.float32), 0.5, B), "clip_tag_")
        for k in ("s30", "s120", "s300"):
            add(grid_mean(z[k].astype(np.float32), 0.5, B), f"clip_{k}")
    e = try_load(lambda: np.load(f"{CUT}/clip_pca.npy", allow_pickle=True).item()[s])
    if e is not None:
        add(grid_mean(e.astype(np.float32), 0.5, B), "clip_pca_")
    e = try_load(lambda: np.load(f"{CUT}/cliptag2.npy", allow_pickle=True).item()[s])
    if e is not None:
        add(grid_mean(e.astype(np.float32), 0.5, B), "clip_tag2_")
    z = try_load(lambda: np.load(f"{CUT}/clap_s{s}.npz"))
    if z is not None:
        add(grid_mean(z["clap"].astype(np.float32), float(z["clap_hop"]), B), "clap_")
    z = try_load(lambda: np.load(f"{CUT}/llm_s{s}.npz"))
    if z is not None:
        add(grid_mean(z["scores"].astype(np.float32), float(z["hop"]), B), "llm_")
    z = try_load(lambda: np.load(f"{CUT}/scene_s{s}.npz", allow_pickle=True))
    if z is not None:
        add(grid_mean(z["scores"].astype(np.float32), float(z["scene"]), B), "scene_")
    e = try_load(lambda: np.load(f"{CUT}/novelty2.npy", allow_pickle=True).item()[s])
    if e is not None:
        add(grid_mean(e.astype(np.float32), 10.0, B), "nov2_")
    z = try_load(lambda: np.load(f"{CUT}/pros_s{s}.npz", allow_pickle=True))
    if z is not None:
        add(grid_mean(z["feats"].astype(np.float32), float(z["grid"]), B), "pros_")
    z = try_load(lambda: np.load(f"{CUT}/surp_s{s}.npz", allow_pickle=True))
    if z is not None:
        add(grid_mean(z["feats"].astype(np.float32), float(z["grid"]), B), "surp_")
    z = try_load(lambda: np.load(f"{CUT}/xvec_s{s}.npz"))
    if z is not None:
        h = float(z["xv_hop"]); xv = z["xv"].astype(np.float32)
        host = xv.mean(0); host /= np.linalg.norm(host) + 1e-6
        add(grid_mean((1 - xv @ host), h, B), "xv_host")
        add(grid_mean(z["dxv"].astype(np.float32), h, B), "xv_d")

    # акустика поэлементно
    src = json.load(open(f"{RULES}/items/s{s}.items.json"))
    st = np.array([r["start"] for r in src]); en = np.array([r["end"] for r in src])
    isw = np.array([r["is_word"] for r in src])
    F = {k: np.array([r[k] for r in src], np.float32)
         for k in ("vol", "vol_std", "zcr", "centroid", "flatness", "voice",
                   "dens5", "dens15", "dens60", "vrel5", "vrel15", "vrel60",
                   "pause_dens", "d_dens", "d_vol")}
    M = np.zeros((n, len(F)), np.float32)
    for k, b in enumerate(B):
        m = (st < b["end"]) & (en > b["start"])
        for j, key in enumerate(F):
            M[k, j] = F[key][m].mean() if m.any() else 0.0
    add(M, "ac_")

    # ---------- типы от языковой модели
    mk = f"{RULES}/marks/s{s}_{a.marks}.json"
    TY = np.zeros((n, len(TYPES)), np.float32)
    FL = np.zeros((n, len(FLAGS)), np.float32)
    if os.path.exists(mk):
        marks = {int(k): v for k, v in json.load(open(mk))["marks"].items()}
        last = None
        for i in range(n):
            v = marks.get(i)
            if v is None and last is not None and B[i]["kind"] == "pause":
                v = last
            if v:
                last = v
                TY[i, TYPES.index(v[0])] = 1
                for f in v[1:]:
                    if f in FLAGS:
                        FL[i, FLAGS.index(f)] = 1
        zl = np.zeros(n, np.float32); zp = np.zeros(n, np.float32)
        zi = np.zeros(n, np.float32)
        cnt = Counter(); i = 0
        while i < n:
            t = int(np.argmax(TY[i])) if TY[i].any() else -1
            j = i
            while j < n and ((int(np.argmax(TY[j])) if TY[j].any() else -1) == t):
                j += 1
            L = dur[i:j].sum(); cnt[t] += 1
            zl[i:j] = L
            zp[i:j] = (mid[i:j] - start[i]) / max(L, 1e-6)
            zi[i:j] = cnt[t]
            i = j
        add(TY, "type_"); add(FL, "flag_")
        add(zl, "zone_len"); add(zp, "zone_pos"); add(zi, "zone_idx")

    # ---------- разности с соседями: прежний конвейер выигрывал от них
    # больше, чем от любого нового блока признаков. Решение зависит не от
    # самой реплики, а от того, чем она отличается от соседних
    A = np.hstack(cols).astype(np.float32)
    base_n = min(A.shape[1], 60)
    Ab = A[:, :base_n]
    for lag in (1, 2, 4, 8, 16):
        back = np.vstack([np.repeat(Ab[:1], lag, 0), Ab[:-lag]]) - Ab
        fwd = np.vstack([Ab[lag:], np.repeat(Ab[-1:], lag, 0)]) - Ab
        add(back, f"lagb{lag}_"); add(fwd, f"lagf{lag}_")

    # ---------- окрестности
    ctx = [roll(dens_char * is_sp, mid, dur, sec) for sec in (30, 120, 600)]
    ctx += [roll(is_sp, mid, dur, sec) for sec in (30, 120, 600)]
    ctx += [roll(nov, mid, dur, sec) for sec in (30, 120, 600)]
    if TY.any():
        for j in range(len(TYPES)):
            ctx += [roll(TY[:, j], mid, dur, sec) for sec in (60, 300)]
    add(np.column_stack(ctx), "ctx_")

    X = np.hstack(cols).astype(np.float32)
    y = np.array([b["used_frac"] for b in B], np.float32)
    os.makedirs(f"{RULES}/feat", exist_ok=True)
    np.savez(f"{RULES}/feat/s{s}.npz", X=X, y=y, dur=dur, start=start,
             kind=is_sp, names=np.array(names, dtype=object))
    print(f"слот {s}: {n} блоков, {X.shape[1]} признаков")


if __name__ == "__main__":
    main()
