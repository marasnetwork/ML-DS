import pandas as pd
import matplotlib.pyplot as plt

used_cars = pd.read_csv("data/vehicles.csv", nrows=100)

print(used_cars[["price", "odometer"]].describe())
print(used_cars[["price", "odometer"]].info())

used_cars = used_cars[(used_cars["price"] > 0) & (used_cars["odometer"] > 0)]
used_cars = used_cars[(used_cars["price"] < 1000000) & (used_cars["odometer"] < 1000000)]
print(used_cars[["price", "odometer"]].describe())

x = used_cars["odometer"]
y = used_cars["price"]

plt.scatter(x, y)
plt.xlabel("Počet ujetých km")
plt.ylabel("Cena $")
plt.title("Bodový graf: cena vs počet ujetých km")
# plt.show()

#Basic prediction
def predict(x, w, b):
  return w*x + b

#Cost J(w, b) functions
def compute_cost(x, y, w, b):
  m = len(y)
  predictions = predict(x, w, b)
  errors = predictions - y
  cost = (1/(2*m))*(errors ** 2).sum()
  return cost

#Gradient descent for appropriate w, b
def gradient_descent_step(x, y, w, b, alpha):
  m = len(y)
  predictions = predict(x, w, b)
  errors = predictions - y

  #Derivation w
  dw = (1/m) * (errors * x).sum()

  #Derivation b
  db = (1/m) * errors.sum()

  w -= alpha * dw
  b -= alpha * db

  return w, b

w = 0
b = 0
alpha = 0.0000000001
num_iters = 1000

#For loop for the best w, b - learning process
for i in range(num_iters):
  w, b = gradient_descent_step(x, y, w, b, alpha)
  if i % 100 == 0:
    cost = compute_cost(x, y, w, b)
    print(f"Iterace {i}: cost = {cost}, w = {w}, b = {b}")

x_vals = x.sort_values()
y_vals = predict(x_vals, w, b)

plt.plot(x_vals, y_vals, color="red", label="Lineární regrese")

#Description
plt.xlabel("Počet ujetých km")
plt.ylabel("Cena $")
plt.title("Regrese: cena vs počet km")
plt.legend()
plt.show()
