#!/usr/bin/env python3
"""
Строит кэши окон сразу для нескольких наборов признаков, читая ленту один раз.
Без этого каждый опыт заново разбирает 138 МБ jsonl.

  prebuild.py 10 ac ac+pca ac+tag ac+vid ac+xv ac+clap ac+pca+tag+vid+xv+clap
"""
import json, collections, sys, os, hashlib
import numpy as np
import windows as M


def main():
    wsec = float(sys.argv[1])
    combos = sys.argv[2:]
    for split in ("train", "test"):
        need = []
        for c in combos:
            blocks = set(c.split("+"))
            key = hashlib.md5(
                f"{split}|{wsec}|{'+'.join(sorted(blocks))}".encode()).hexdigest()[:12]
            p = f"{M.CACHE}/w_{key}.npz"
            if not os.path.exists(p):
                need.append((blocks, p, c))
        if not need:
            continue
        os.makedirs(M.CACHE, exist_ok=True)
        rows = [json.loads(l) for l in open(f"{M.DATA}/{split}.jsonl")]
        if split == "test":
            ans = [json.loads(l) for l in open(f"{M.DATA}/test_answers.jsonl")]
            for r, a in zip(rows, ans):
                r["used_frac"] = a["used_frac"]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["slot"]].append(r)
        for blocks, p, c in need:
            X, D = {}, {}
            for s in sorted(by):
                E = M.ext_features(s, by[s], blocks)
                F, Y, WT, nac = M.build_windows(by[s], E, wsec)
                X[s] = M.context(F, wsec, nac, qn="qn" in blocks); D[s] = (Y, WT)
            np.savez(p, X=X, D=D)
            print(f"{split} {c}: {X[sorted(by)[0]].shape[1]} признаков -> {p}",
                  flush=True)


if __name__ == "__main__":
    main()
