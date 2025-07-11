####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 8.
# Pandas: Datetimeindex, resampling
####################################

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download("VUAA.DE", start="2025-01-01", end="2025-07-02")

# 20denní klouzavý průměr
df["MA20"] = df["Close"].rolling(window=20).mean()

# Výpočet procentuální změny mezi začátkem a koncem
start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]
pct_change = ((end_price - start_price) / start_price) * 100
pct_change = round(pct_change, 2)

plt.figure(figsize=(14, 6))
plt.plot(df["Close"], label=f"Zavírací cena ({pct_change}%)")
plt.plot(df["MA20"], label="20denní klouzavý průměr")
plt.title("VUAA.DE - cena")
plt.xlabel("Datum")
plt.ylabel("Cena (EUR)")
plt.legend()
plt.grid(True)
plt.show()
