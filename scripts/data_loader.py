import os
import pandas as pd


def load_csv(path):
    return pd.read_csv(path)

def load_json(path):
    return pd.read_json(path)

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return pd.DataFrame(lines, columns=["TextData"])

def load_excel(path):
    return pd.read_excel(path)

