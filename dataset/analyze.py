#!/usr/bin/env python3
"""Разбор обучающей выборки: что там есть и что стоит поправить."""
import os
import json, collections, statistics as st

OUT = os.environ.get("SEM_DATASET", "work/dataset")
d = json.load(open(f"{OUT}/stream_2026-06-30_varenikovskiy.labeled.json"))
I = d["items"]
W = [i for i in I if i["type"] == "word"]
S = [i for i in I if i["type"] == "silence"]
DUR = d["duration_sec"]


def p(t):
    print(f"\n{'='*62}\n{t}\n{'='*62}")


def q(v, name):
    v = sorted(v)
    n = len(v)
    print(f"  {name}: n={n} мин={v[0]:.2f} p25={v[n//4]:.2f} мед={v[n//2]:.2f} "
          f"p75={v[3*n//4]:.2f} p95={v[int(n*.95)]:.2f} макс={v[-1]:.2f}")


p("1. БАЛАНС КЛАССОВ")
wu = sum(i["used"] for i in W)
su = sum(i["used"] for i in S)
print(f"  слова:  used {wu} / {len(W)} = {wu/len(W)*100:.1f}%")
print(f"  тишина: used {su} / {len(S)} = {su/len(S)*100:.1f}%")
print(f"  всего:  used {wu+su} / {len(I)} = {(wu+su)/len(I)*100:.1f}%")
print(f"  -> классы почти сбалансированы, перевес 60/40 в сторону 'выброшено'")

p("2. ДЛИТЕЛЬНОСТИ")
q([i["end"]-i["start"] for i in W], "слова      ")
q([i["end"]-i["start"] for i in S], "тишина     ")
q([i["end"]-i["start"] for i in W if i["used"]], "слова used ")
q([i["end"]-i["start"] for i in W if not i["used"]], "слова drop ")
q([i["end"]-i["start"] for i in S if i["used"]], "тишина used")
q([i["end"]-i["start"] for i in S if not i["used"]], "тишина drop")

p("3. ДЛИНА ТИШИНЫ КАК ПРИЗНАК")
buckets = [(0,.15),(.15,.3),(.3,.6),(.6,1),(1,2),(2,5),(5,99)]
for a,b in buckets:
    g=[i for i in S if a<=i["end"]-i["start"]<b]
    if g:
        print(f"  тишина {a:>4}-{b:<4}с: {len(g):>5} шт, оставлено {sum(i['used'] for i in g)/len(g)*100:5.1f}%")

p("4. ОЧЕНЬ КОРОТКИЕ ЭЛЕМЕНТЫ")
tiny_w=[i for i in W if i["end"]-i["start"]<=0.06]
tiny_s=[i for i in S if i["end"]-i["start"]<=0.06]
print(f"  слова <=0.06 с: {len(tiny_w)} ({len(tiny_w)/len(W)*100:.1f}%)")
print(f"  тишина <=0.06 с: {len(tiny_s)} ({len(tiny_s)/len(S)*100:.1f}%)")
print(f"  примеры слов: {', '.join(repr(i['word']) for i in tiny_w[:8])}")

p("5. ДЛИННЫЕ СЛОВА БЕЗ ТИШИНЫ ВНУТРИ (подозрение на шум)")
long_w=[i for i in W if i["end"]-i["start"]>4]
print(f"  слов длиннее 4 с: {len(long_w)}")
for i in sorted(long_w,key=lambda x:-(x['end']-x['start']))[:8]:
    print(f"    {i['end']-i['start']:5.2f}с  {i['start']/60:7.1f}мин  used={str(i['used']):5}  {i['word'][:30]}")

p("6. СТРУКТУРА: НЕПРЕРЫВНЫЕ УЧАСТКИ")
runs=[]; cur=I[0]["used"]; n=1; t=I[0]["end"]-I[0]["start"]
for i in I[1:]:
    if i["used"]==cur: n+=1; t+=i["end"]-i["start"]
    else: runs.append((cur,n,t)); cur=i["used"]; n=1; t=i["end"]-i["start"]
runs.append((cur,n,t))
ru=[r for r in runs if r[0]]; rd=[r for r in runs if not r[0]]
print(f"  участков 'оставлено': {len(ru)}, медиана {st.median(r[2] for r in ru):.1f} с, "
      f"макс {max(r[2] for r in ru):.0f} с")
print(f"  участков 'выброшено': {len(rd)}, медиана {st.median(r[2] for r in rd):.1f} с, "
      f"макс {max(r[2] for r in rd):.0f} с")
print(f"  -> модель должна предсказывать не отдельные элементы, а ПЕРЕХОДЫ:")
print(f"     их всего {len(runs)-1} на {len(I)} элементов = {(len(runs)-1)/len(I)*100:.1f}%")

p("7. ПОЗИЦИЯ В СТРИМЕ")
for k in range(10):
    a,b=DUR*k/10, DUR*(k+1)/10
    g=[i for i in I if a<=i["start"]<b]
    if g:
        bar="#"*int(sum(i['used'] for i in g)/len(g)*40)
        print(f"  {a/60:5.0f}-{b/60:3.0f} мин  {sum(i['used'] for i in g)/len(g)*100:5.1f}%  {bar}")

p("8. СЛОВАРЬ")
words=[i["word"].strip('.,!?»«:;()').lower() for i in W]
c=collections.Counter(words)
print(f"  уникальных словоформ: {len(c)}, всего слов {len(words)}")
print(f"  топ-12 и доля оставленных:")
for w,n in c.most_common(12):
    g=[i for i in W if i["word"].strip('.,!?»«:;()').lower()==w]
    print(f"    {w:<12} {n:>4} раз, оставлено {sum(i['used'] for i in g)/len(g)*100:5.1f}%")
hapax=sum(1 for w,n in c.items() if n==1)
print(f"  встречаются один раз: {hapax} ({hapax/len(c)*100:.0f}% словаря)")

p("9. ГРАНИЦЫ СКЛЕЕК")
edge=0
for k,i in enumerate(I[1:-1],1):
    if i["type"]=="word" and i["used"] and (not I[k-1]["used"] or not I[k+1]["used"]):
        edge+=1
print(f"  слов на краю склейки: {edge} ({edge/wu*100:.1f}% от оставленных)")
print(f"  их метка ненадёжна: слово могло войти в выпуск обрезанным")
