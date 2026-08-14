#!/usr/bin/env python3
"""
Восстановление монтажного листа: какие куски полного стрима попали в ролик.

Стратегия:
  1. глобальный поиск — где в стриме звучит начало очередного фрагмента ролика;
  2. ведение — пока сдвиг держится, идём вперёд дешёвыми локальными проверками;
  3. срыв соответствия = склейка, точное место ищем двоичным поиском;
  4. если ничего не нашлось (реклама, вставки) — помечаем как «не из стрима».
"""
import json, os, sys
import numpy as np

SP = os.environ.get("SEM_EDL_WORK", os.path.dirname(os.path.abspath(__file__)))  # промежуточные файлы
FPS = 100.0
SR = 8000

W_GLOBAL = 400      # окно глобального поиска, кадров (4 с)
W_VERIFY = 60       # окно локальной проверки (0.6 с)
NCC_GLOBAL = 0.30   # порог принятия глобального совпадения
NCC_VERIFY = 0.25   # порог, что соответствие ещё держится
STEP = 100          # шаг ведения, кадров (1 с)
MIN_SEG = 20        # короче 0.2 с фрагменты не рассматриваем


class Matcher:
    def __init__(self, stream_env, edit_env):
        self.S = stream_env.astype(np.float64)
        self.E = edit_env.astype(np.float64)
        self.N = len(self.S)
        # накопительные суммы для нормировки окон стрима
        self.cs = np.concatenate(([0.0], np.cumsum(self.S)))
        self.cs2 = np.concatenate(([0.0], np.cumsum(self.S ** 2)))
        self.nfft = 1 << (self.N + W_GLOBAL).bit_length()
        print(f"  стрим {self.N} кадров, ролик {len(self.E)} кадров, FFT {self.nfft}")
        self.FS = np.fft.rfft(self.S, self.nfft)

    def _win_stats(self, w):
        """среднее и норма центрированного окна длины w для всех позиций"""
        s = self.cs[w:] - self.cs[:-w]
        s2 = self.cs2[w:] - self.cs2[:-w]
        var = np.maximum(s2 - s * s / w, 1e-9)
        return s / w, np.sqrt(var)

    def global_search(self, q):
        """где в стриме встречается q. -> (позиция кадра, NCC)"""
        w = len(q)
        q = q - q.mean()
        nq = np.linalg.norm(q)
        if nq < 1e-9:
            return -1, 0.0
        q = q / nq
        FQ = np.fft.rfft(q, self.nfft)
        corr = np.fft.irfft(self.FS * np.conj(FQ), self.nfft)[:self.N - w + 1]
        _, sd = self._win_stats(w)
        ncc = corr / sd
        k = int(np.argmax(ncc))
        return k, float(ncc[k])

    def verify(self, te, ts, w=W_VERIFY):
        """совпадение ролика с te и стрима с ts при нулевом сдвиге"""
        if te < 0 or ts < 0 or te + w > len(self.E) or ts + w > self.N:
            return -1.0
        a = self.E[te:te + w]
        b = self.S[ts:ts + w]
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return -1.0
        return float(np.dot(a, b) / (na * nb))


def run():
    S = np.fromfile(os.path.join(SP, "stream.env"), dtype=np.float32)
    E = np.fromfile(os.path.join(SP, "edit.env"), dtype=np.float32)
    m = Matcher(S, E)
    M = len(E)

    segments, unmatched = [], []
    t = 0
    n_global = 0
    while t < M - MIN_SEG:
        w = min(W_GLOBAL, M - t)
        if w < MIN_SEG:
            break
        k, ncc = m.global_search(E[t:t + w])
        n_global += 1
        if ncc < NCC_GLOBAL:
            # не нашли — двигаемся вперёд, копим «чужой» участок
            if unmatched and unmatched[-1][1] == t:
                unmatched[-1] = (unmatched[-1][0], t + 50)
            else:
                unmatched.append((t, t + 50))
            t += 50
            continue

        d = k - t                      # сдвиг: стрим = ролик + d
        start_e = t
        # ведём вперёд
        x = t
        while True:
            nxt = x + STEP
            if nxt + W_VERIFY > M or nxt + d + W_VERIFY > m.N:
                x = min(M - W_VERIFY, x)
                break
            if m.verify(nxt, nxt + d) >= NCC_VERIFY:
                x = nxt
            else:
                break
        # двоичный поиск точного конца между x и x+STEP
        lo, hi = x, min(x + STEP, M - W_VERIFY)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if m.verify(mid, mid + d) >= NCC_VERIFY:
                lo = mid
            else:
                hi = mid
        end_e = lo + W_VERIFY          # конец фрагмента в координатах ролика
        end_e = min(end_e, M)
        if end_e - start_e >= MIN_SEG:
            segments.append({"e0": start_e, "e1": end_e, "d": d, "ncc": ncc})
            t = end_e
        else:
            t += 50

        if len(segments) % 50 == 0 and segments:
            print(f"    найдено {len(segments)} фрагментов, "
                  f"дошли до {t/FPS/60:.1f} мин из {M/FPS/60:.1f}")

    print(f"\n  глобальных поисков: {n_global}")
    print(f"  фрагментов: {len(segments)}")
    print(f"  участков без совпадения: {len(unmatched)}")
    json.dump({"fps": FPS, "segments": segments, "unmatched": unmatched},
              open(os.path.join(SP, "raw_match.json"), "w"), indent=1)
    return segments, unmatched


if __name__ == "__main__":
    run()
