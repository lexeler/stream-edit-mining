#!/usr/bin/env python3
"""
Расширенный набор вопросов к кадру. Векторы CLIP уже посчитаны, так что это
просто перемножение матриц — секунды вместо часов.

Два изменения против первой версии:
  вопросов 64 вместо 24, покрывают быт уличного стрима подробнее;
  добавлены ПРОТИВОПОСТАВЛЕНИЯ — пары «скучно/интересно», «один/с людьми».
  CLIP плохо отвечает на вопрос в лоб, но хорошо сравнивает два описания,
  поэтому разность близостей внутри пары информативнее самой близости.
"""
import os
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor

DATA = os.path.join(os.environ.get("SEM_WORK", "work/window"), "data")
MODEL = "openai/clip-vit-large-patch14"
SLOTS = ["1", "2", "3", "4", "5", "6", "7"]

TAGS = [
    "a group of friends hanging out together",
    "a crowd of people in the background",
    "two people talking face to face",
    "a single person alone with nobody around",
    "a close-up of a smiling laughing face",
    "a person looking bored or tired",
    "a man holding a camera and talking to it",
    "a table with food and drinks, people eating",
    "the interior of a cafe or restaurant",
    "a bar or nightclub with lights",
    "a shop or supermarket aisle",
    "the inside of a car, driving on a road",
    "a view through a car windshield of an empty road",
    "a passenger seat view inside a moving vehicle",
    "walking along a sidewalk in a city",
    "an empty street with nobody around",
    "a busy street full of people",
    "a park with trees and grass",
    "an abandoned ruined building",
    "a construction site",
    "a beach, a river or the sea",
    "a mountain or countryside landscape",
    "a residential courtyard with apartment blocks",
    "the inside of an apartment or a room",
    "a hotel room or hostel",
    "a staircase or a corridor",
    "public transport, a bus or a train",
    "a gas station or a parking lot",
    "a market or street stalls",
    "a sports field or gym",
    "a dog or a cat",
    "a phone screen being shown to the camera",
    "a laptop or computer monitor",
    "money, banknotes or a wallet",
    "alcohol bottles or beer",
    "a plate of food close up",
    "night time with street lights",
    "bright sunny daylight",
    "rain or wet weather",
    "snow or winter",
    "a very dark almost black image",
    "an overexposed washed out image",
    "a blurry motion-smeared image",
    "the camera pointing at the ground",
    "the camera pointing at the sky",
    "a shaky handheld shot while walking",
    "a steady static shot",
    "an unusual or surprising situation",
    "a fight, argument or conflict",
    "someone performing or showing a trick",
    "people playing a game together",
    "a person swimming or in water",
    "someone climbing or jumping",
    "an accident or something broken",
    "police or security officers",
    "a queue of people waiting",
    "someone sleeping or lying down",
    "an empty room with nothing happening",
    "a scene where nothing interesting is going on",
    "an exciting moment worth watching",
    "a funny moment, people cracking up",
    "an awkward silence between people",
    "a person explaining something with gestures",
    "a stranger being approached on the street",
]

# пары для противопоставления: индексы (положительный, отрицательный)
PAIRS = [(0, 3), (4, 5), (16, 15), (59, 58), (60, 61), (47, 57), (7, 12)]


@torch.no_grad()
def main():
    dev = "cuda:0"
    m = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    pr = CLIPProcessor.from_pretrained(MODEL)
    t = pr(text=TAGS, return_tensors="pt", padding=True)
    T = m.get_text_features(**{k: v.to(dev) for k, v in t.items()})
    T = (T / T.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    print(f"вопросов {len(TAGS)}, пар противопоставления {len(PAIRS)}")

    out = {}
    for s in SLOTS:
        e = np.load(f"{DATA}/clip_s{s}.npz")["emb"].astype(np.float32)
        sc = e @ T.T
        # внутри стрима вычитаем среднее: абсолютные близости CLIP сжаты
        sc = (sc - sc.mean(0)) / (sc.std(0) + 1e-6)
        pairs = np.stack([sc[:, a] - sc[:, b] for a, b in PAIRS], 1)
        out[s] = np.hstack([sc, pairs]).astype(np.float32)
        print(f"слот {s}: {out[s].shape}")
    np.save(f"{DATA}/cliptag2.npy", out, allow_pickle=True)
    print("сохранено")


if __name__ == "__main__":
    main()
