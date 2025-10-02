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

#Task 1: Load wine data
def load_wine_data(folder_path):
    
    red_path = os.path.join(folder_path, "winequality-red.xlsx")
    white_path = os.path.join(folder_path, "winequality-white.xlsx")
    
    #Task 4: Load the public dataset
    #https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download
    #Where the public data is from
    internet_wine = os.path.join(folder_path, "WineQT.csv")

    red_wine = pd.read_excel(red_path, header=1)
    white_wine = pd.read_excel(white_path, header=1)
    internet_wine = pd.read_csv(internet_wine)

    #Task 2: Clean the data
    red_wine.columns = red_wine.columns.str.lower().str.replace(" ", "_")
    red_wine["type"] = "Red Wine"

    white_wine.columns = white_wine.columns.str.lower().str.replace(" ", "_")
    white_wine["type"] = "White Wine"

    internet_wine.columns = internet_wine.columns.str.lower().str.replace(" ", "_")
    if 'id' in internet_wine.columns:
        internet_wine = internet_wine.drop(columns=['id'])
    internet_wine["type"] = "Wine from public sources"

    #Task 3: Combine the datasets
    df = pd.concat([red_wine, white_wine, internet_wine], ignore_index=True)
    df = df.drop_duplicates()

    return df
