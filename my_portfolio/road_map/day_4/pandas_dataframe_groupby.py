####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 4.
# Pandas: DataFrame, groupby, aggregation
####################################

import pandas as pd

passengers = pd.DataFrame({
    "PassengerId": [1, 2, 3],
    "Name": ["John", "Anna", "Peter"]
})

ages = pd.DataFrame({
    "PassengerId": [1, 2, 3],
    "Age": [22, 30, 19]
})

# Merge both tables
merged_df = pd.merge(passengers, ages, on="PassengerId")
print(merged_df)

df = pd.read_csv("../data/train.csv")

mean_age_by_class = df.groupby("Pclass")["Age"].mean()
print()
print("Mean age by Pclass")
print(mean_age_by_class)

# Mean price by pclass
mean_price_by_class = df.groupby("Pclass")["Fare"].mean()
print()
print("Mean price by pclass")
print(mean_price_by_class)

# Mean survived by class
mean_survived_by_class = df.groupby("Pclass")["Survived"].mean()*100
print()
print("Mean survived by class")
print(mean_survived_by_class)
