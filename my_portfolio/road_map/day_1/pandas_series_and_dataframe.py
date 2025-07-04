####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 1.
# Pandas: Series and DataFrame
####################################

import pandas as pd # type: ignore

df = pd.read_csv("train.csv")

print("First five rows")
print(df.head())
print()

print("Name of columns")
print(df.columns)
print()

print("Age column")
print(df["Age"].head())
print()

print("Multiple columns")
print(df[["Age", "Survived"]].head())
print()

print("Basic information about columns")
print(df.info())
print()

print("Basic statistics")
print(df.describe())
print()

print("How many people survived")
print(df["Survived"].value_counts())
print()

print("How many unique values are in column")
print(df["Embarked"].unique())
print()

print("Missing values in column")
print(df["Age"].isnull().sum())
print()

print("Passengers count")
print(len(df))
print()

print("Men and women count")
print(df["Sex"].value_counts())
print()

print("Average age of survivors")
print(df[df["Survived"] == 1]["Age"].mean())
print()

print("Average price of ticket")
print(df.groupby("Pclass")["Fare"].mean())
print()

print("Missing age values")
print(df["Age"].isnull().sum())
print()

print("Passengers between 18-25 in first class")
print(len(df[(df["Age"] >= 18) & (df["Age"] <= 25) & (df["Pclass"] == 1)]))
print()

print("The most expensive ticker")
print(df["Fare"].max())
print()

print("The count of people in every seaport")
print(df["Embarked"].value_counts())
print()
