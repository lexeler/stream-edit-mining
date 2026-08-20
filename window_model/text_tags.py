#!/usr/bin/env python3
"""
Смысловые метки текста поверх УЖЕ посчитанных векторов e5 (sem_raw.npy).

Сжатие 1024 -> 48 дало мало (+5,4 пункта). Пробуем иначе: вместо безымянных
компонент задаём окну прямые вопросы и берём близость. Получается два десятка
понятных признаков вместо сотни шумных — при 2400 решениях это важнее.

e5 обучена на парах query/passage, окна были закодированы как passage,
поэтому вопросы кодируем как query.
"""
import os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "intfloat/multilingual-e5-large"

TAGS = {
    "смех":        "люди смеются, шутят, ржут",
    "история":     "рассказывает историю, случай из жизни",
    "спор":        "спорят, ругаются, конфликт, наезд",
    "событие":     "неожиданно что-то произошло, реакция на событие",
    "незнакомец":  "разговор с незнакомым человеком на улице",
    "игра":        "игра, соревнование, спор на деньги, челлендж",
    "еда":         "еда, кафе, заказывают, едят и пьют",
    "деньги":      "деньги, сколько стоит, цены, покупка",
    "дорога":      "едут в машине, за рулём, дорога",
    "идут":        "идут пешком, ищут дорогу, куда идти",
    "тишина":      "молчание, ничего не происходит, пауза",
    "техника":     "проблемы со стримом, связь, звук, батарея, камера",
    "чат":         "читает чат, отвечает зрителям, донат, подписка",
    "ниочём":      "скучный однообразный разговор ни о чём",
    "паразиты":    "ну это, короче, э-э-э, мычание и запинки",
    "привет":      "приветствие, здорово, всем привет, начало стрима",
    "пока":        "прощание, всем пока, конец стрима, до завтра",
    "мат":         "мат, ругательства, грубые слова",
    "вопрос":      "задаёт вопрос собеседнику",
    "эмоция":      "кричит, орёт, сильные эмоции, восторг",
    "объясняет":   "объясняет, рассказывает как что устроено",
    "планы":       "обсуждают планы, что будем делать дальше",
}


@torch.no_grad()
def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval().to(dev).half()
    names = list(TAGS)
    b = tok(["query: " + TAGS[k] for k in names], return_tensors="pt",
            padding=True, truncation=True, max_length=64).to(dev)
    o = mdl(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1)
    Q = torch.nn.functional.normalize((o * m).sum(1) / m.sum(1), dim=-1)
    Q = Q.float().cpu().numpy()
    print(f"вопросов: {len(names)}")

    store = np.load(f"{DATA}/sem_raw.npy", allow_pickle=True).item()
    out = {}
    for s, d in store.items():
        out[s] = {"idx": d["idx"], "names": names}
        for sc in ("near", "far"):
            out[s][sc] = (d["emb"][sc] @ Q.T).astype(np.float32)
        print(f"слот {s}: {out[s]['near'].shape}")
    np.save(f"{DATA}/semtags.npy", out, allow_pickle=True)

    # диагностика: расходятся ли метки между оставленным и выброшенным
    import json, collections
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    print(f"\n{'метка':>12}{'оставлено':>12}{'выброшено':>12}{'разница':>10}")
    print("-" * 46)
    diffs = []
    for j, nm in enumerate(names):
        a = b_ = aw = bw = 0.0
        for s, rr in by.items():
            v = out[s]["near"][out[s]["idx"], j]
            f = np.array([r["used_frac"] for r in rr], np.float32)
            w = np.array([r["dur"] for r in rr], np.float32)
            a += float((v * f * w).sum()); aw += float((f * w).sum())
            b_ += float((v * (1 - f) * w).sum()); bw += float(((1 - f) * w).sum())
        diffs.append((a / aw - b_ / bw, nm, a / aw, b_ / bw))
    for d, nm, x, y in sorted(diffs, key=lambda z: -abs(z[0])):
        print(f"{nm:>12}{x:>12.4f}{y:>12.4f}{d:>+10.4f}")


if __name__ == "__main__":
    main()
