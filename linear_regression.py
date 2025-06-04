import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./deeplearning.mplstyle")

#x_train - input (size in 1000 square feet)
x_train = np.array([1.0, 2.0])

#y_train - target (price in 1000s of dollars)
y_train = np.array([300.00, 500.00])

#Basic output
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
print()

#Length of the array
print(f"x_train.shape: {x_train.shape}")
print()

m = x_train.shape[0]

print(f"Počet trénovacích příkladů: {m}")
print()

#Access to value in the array with index
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
print()

#Plotting the data
#Plot the data points
plt.scatter(x_train, y_train, marker="x", c="r")
#Set the title
plt.title("Ceny domů")
#Set the y-axis label
plt.ylabel("Cena (v tisících dollarů)")
#Set the x-axis label
plt.xlabel("Velikost (1000 sqft)")
# plt.show()

#Model parameters
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")
print()

#Compute value
# f_wb = w * x_i + b

def compute_model_output(x, w, b):

  m = x.shape[0]

  #Create a vector with length m, where all values are 0
  f_wb = np.zeros(m)

  for i in range(m):
    print(f_wb)
    f_wb[i] = w * x[i] + b
    print(f_wb)

  return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

#Test with unique x
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} tisíc dollarů")

#Plot our model predictions
plt.plot(x_train, tmp_f_wb, c="b", label="Naše predikce")

#Plot the data points
plt.scatter(x_train, y_train, marker="x", label="Reálné hodnoty", c="r")

plt.show()