####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 6.
# Pandas: Pandas, null values
####################################

import pandas as pd

df = pd.read_csv("../data/train.csv")

# Počet chybějících hodnot v každém sloupci
print(df.isnull().sum())

# Konkrétní řádky obsahující hodnotu null
print(df[df["Age"].isnull()])
print(f"Prázdné hodnoty: {df["Age"].isnull().sum()}")

# Nahrazení konkrétní hodnotou
# df["Age"] = df["Age"].fillna(df["Age"].mean())

# Smazání řádků s chybějícími hodnotami
# df.dropna(inplace=True)

average_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(average_age)

most_common = df["Embarked"].value_counts().idxmax()
df["Embarked"] = df["Embarked"].fillna(most_common)
