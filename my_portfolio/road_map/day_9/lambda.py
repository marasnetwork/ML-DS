####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 9.
# Lambda, apply functions
####################################

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

df = pd.read_csv("../data/train.csv")

def categorize_age(age):
    if pd.isnull(age):
        return "Unknown"
    elif age < 18:
        return "Mladý"
    elif age < 60:
        return "Dospělý"
    else:
        return "Senior"
    
df["AgeGroup"] = df["Age"].apply(lambda x: categorize_age(x))

print(df[["Age", "AgeGroup"]].head(10))
print(df["AgeGroup"].value_counts())
print("-"*100)

# 2. project - Xtrackers Nasdaq 100

def daily_trend(row):
    close = row[('Close', 'XNAS.L')]
    prev_close = row[("Previous close", "")]
    if pd.isna(close) or pd.isna(prev_close):
        return "None"
    if close > prev_close:
        return "UP"
    elif close < prev_close:
        return "DOWN"
    else:
        return "SAME"

da = yf.download("XNAS.L", start="2024-01-01", end="2025-07-01")
da_info = yf.Ticker("XNAS.L").info

da["Previous close"] = da["Close"].shift(1)

print("-"*100)
print(da)
print("-"*100)
print(da.iloc[0])

da["Trend"] = da.apply(lambda row: daily_trend(row), axis=1)

# 20denní klouzavý průměr
da["MA20"] = da["Close"].rolling(window=20).mean()

# print(da_info["currency"])

plt.figure(figsize=(14, 6))
plt.plot(da["Close"], label="Zavírací cena")
plt.plot(da["MA20"], label="20denní klouzavý průměr")
plt.title("Xtrackers Nasdaq 100 - cena")
plt.xlabel("Čas")
plt.ylabel("Cena")
plt.grid(True)
plt.show()

print("-"*100)
print(da["Trend"].value_counts())
print("-"*100)
