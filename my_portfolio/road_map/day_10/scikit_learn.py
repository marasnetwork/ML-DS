####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 10.
# Scikit learn, StandardScaler
####################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/train.csv")

features = ["Pclass", "Sex", "Age", "Fare"]
target = "Survived"

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df = df["Age"].dropna()
