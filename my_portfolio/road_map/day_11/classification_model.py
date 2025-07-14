####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 11.
# Classification model
####################################

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

apple = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Vezmi jen Close ceny
apple = apple[["Close"]].copy()

# Vytvoř sloupec "Tomorrow" (zítřejší cena)
apple["Tomorrow"] = apple["Close"].shift(-1)

# Odstraň řádky s chybějícími hodnotami
apple.dropna(inplace=True)

# Vytvoř cílovou proměnnou: 1 pokud zítra cena vzroste, jinak 0
apple["Target"] = (apple["Tomorrow"].squeeze() > apple["Close"].squeeze()).astype(int)

# Vstupní a výstupní proměnné
X = apple[["Close"]]
y = apple["Target"]

# Rozdělení na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Škálování
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistická regrese
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Předpověď
y_pred = model.predict(X_test_scaled)

# Vyhodnocení
print("Přesnost:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))