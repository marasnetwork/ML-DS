####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 10.
# Scikit learn, StandardScaler
####################################

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

btc = yf.download("BTC-USD", start="2022-01-01", end="2025-01-01")

print("-"*120)
print(btc.head())
print("-"*120)

btc["Close"].plot(figsize=(14, 5), title="Bitcoin zavírací cena")
plt.xlabel("Datum")
plt.ylabel("Cena v USD")
plt.grid(True)
plt.show()

# Split dat na trénovací a testovací sadu
btc = btc[["Close"]].dropna() # Odstraní řádky s chybějícími hodnotami

btc["Tomorrow"] = btc["Close"].shift(-1)
btc = btc.dropna()

X = btc[["Close"]]
y = btc["Tomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalizace dat (škálování)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vytvoření modelu a natrénování
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(14, 5))
plt.plot(y_test.values, label="Sktuečné ceny")
plt.plot(y_pred, label="Predikované ceny")
plt.title("Predikce ceny Bitcoinu")
plt.xlabel("Dny")
plt.ylabel("Cena v USD")
plt.legend()
plt.grid(True)
plt.show()

