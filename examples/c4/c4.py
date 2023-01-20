""""
Convert the C4 dataset from JSONL file to a balanced dataset with randomly 
generated negative samples and save it as a CSV file.

Original JSONL file: https://github.com/Chenning-Tao/C4/tree/main/dataset

Usage: python c4.py
(This will replace the existing c4.csv file)
"""
import json
import random
import pandas as pd
from pathlib import Path

codes = set()
clones = {}
with open(Path(__file__).parent / "c4.jsonl", "r") as f:
    for line in f.readlines():
        dic = json.loads(line)
        if dic["Category1"] == "py":
            codes.add(dic["Code1"])
        if dic["Category2"] == "py":
            codes.add(dic["Code2"])
        if dic["Category1"] == "py" and dic["Category2"] == "py":
            clones.setdefault(dic["Code1"], {dic["Code1"]}).add(dic["Code2"])

positive = set()
negative = set()
for key in clones:
    clone_pairs = set(
        [(key, value, 1) for value in clones[key] if (value, key, 1) not in positive]
    )
    positive |= clone_pairs

    negative |= set(
        [
            (key, value, 0)
            for value in random.sample(tuple(codes - clones[key]), len(clone_pairs))
        ]
    )

df = pd.DataFrame((positive | negative), columns=["code1", "code2", "similar"])
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("c4.csv", index=False)
