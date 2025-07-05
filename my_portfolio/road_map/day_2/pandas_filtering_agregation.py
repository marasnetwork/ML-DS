####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 2.
# Pandas: Filtering, agregation
####################################

import pandas as pd # type: ignore

df = pd.read_csv("../data/train.csv")

print("Average age survived")
average_age_survived = df[df["Survived"] == 1]["Age"].mean()
print(average_age_survived)
print()

print("Average age not survived")
average_age_not_survived = df[df["Survived"] == 0]["Age"].mean()
print(average_age_not_survived)
print()

print("Men and women count")
men_and_women_count = df["Sex"].value_counts()
print(men_and_women_count)
print()

print("Men and women survived count")
men_and_women_survived_count = df[df["Survived"] == 1]["Sex"].value_counts()
print(men_and_women_survived_count)
print()

print("Women in first class survived")
women_in_first_class_survived_count = df[(df["Sex"] == "female") & (df["Survived"] == 1) & (df["Pclass"] == 1)].shape[0]
print(women_in_first_class_survived_count)
print()

print("Count of survivors")
survivors_count = df["Survived"].sum()
print(survivors_count)
print()

print("Age not NaN")
age_not_nan = df["Age"].count()
print(age_not_nan)
print()

print("Men ticket > 50")
men_ticket_50 = df[(df["Sex"] == "male") & (df["Fare"] > 50)].shape[0]
print(men_ticket_50)
print()

print("Women count in first class with price < 80")
women_count_first_class_ticket_80_less = df[(df["Sex"] == "female") & (df["Fare"] < 80) & (df["Pclass"] == 1)].shape[0]
print(women_count_first_class_ticket_80_less)
print()
