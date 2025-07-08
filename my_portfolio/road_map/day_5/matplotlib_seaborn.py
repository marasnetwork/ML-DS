####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 5.
# Pandas: Matplotlib, seaborn
####################################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("../data/train.csv")

# plt.hist(df["Age"].dropna(), bins=30, color="skyblue", edgecolor="black")
sns.histplot(df["Age"], bins=30, kde=True, color="coral")
plt.title("Histogram věku pasažérů")
plt.xlabel("Věk")
plt.ylabel("Počet lidí")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Age", y="Fare", data=df, hue="Survived", palette="coolwarm", alpha=0.6)
plt.title("Scatter plot: věk vs cena lístku")
plt.xlabel("Věk")
plt.ylabel("Cena lístku")
plt.grid(True)
plt.show()
