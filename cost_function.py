import math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use("./deeplearning.mplstyle")

#Description of this project
#Model which can predict housing prices given the size of the house
#Info: This project is inspired by Andrew Ng labs

#Training data
x_train = np.array([1.0, 2.0]) #size in 1000 square feet
y_train = np.array([300.00, 500.00]) #price in 1000s of dollars

#Cost is a measure how well our model is predicting the target price of the houses

#Computes the cost function for linear regression
def compute_cost(x, y, w, b):
  m = x_train.shape[0]

  cost_sum = 0
  
  for i in range(m):
    f_wb = w * x[i] + b
    cost = math.pow(f_wb - y[i], 2)
    cost_sum += cost

  return (1 / (2 * m)) * cost_sum

w = 200
b = 100
cost = compute_cost(x_train, y_train, w, b)
print(f"Cost: {cost}")

plt_intuition(x_train, y_train)
