import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_gradients, plt_update_onclick, soup_bowl, plt_contour_wgrad
plt.style.use("./deeplearning.mplstyle")

#Description of this project
#Model which can predict housing prices given the size of the house
#Info: This project is inspired by Andrew Ng labs

#Data set
x_train = np.array([1.0, 2.0]) #size in 1000 square feet
y_train = np.array([300.0, 500.0]) #price in 1000s of dollars

def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost = 0

  for i in range(m):
    f_wb = w * x[i] + b
    cost += math.pow(f_wb - y[i], 2)
  return 1 / (2 * m) * cost

#Gradient function
def compute_gradient(x, y, w, b):
  
  m = x.shape[0]
  dj_dw = 0
  dj_db = 0

  for i in range(m):
    f_wb = w * x[i] + b
    prediction = (f_wb - y[i])
    dj_dw_i = prediction * x[i]
    dj_db_i = prediction
    dj_dw += dj_dw_i
    dj_db += dj_db_i
    
  dj_dw = dj_dw / m
  dj_db = dj_db / m

  return dj_dw, dj_db

#Gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
  w = copy.deepcopy(w_in)

  #An array to store cost J and w's at each iteration primarily for graphing later
  J_history = []
  p_history = []
  b = b_in
  w = w_in

  for i in range(num_iters):
    #Calculate the gradient and update the parameters using gradient_function
    dj_dw, dj_db = gradient_function(x, y, w, b)

    #Update parameters
    b -= alpha * dj_db
    w -= alpha * dj_dw

    #Save cost J at each iteration
    if i < 100000:
      J_history.append(cost_function(x, y, w, b))
      p_history.append([w, b])
    #Print cost every at intervals 10 times or as many iterations if < 10
    if i% math.ceil(num_iters/10) == 0:
      print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ", f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ", f"w: {w: 0.3e}, b:{b: 0.5e}")
      
  return w, b, J_history, p_history #return w and J,w history for graphing

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

plt_gradients(x_train, y_train, compute_cost, compute_gradient)


#Plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5], contours=[1,5,10,20],resolution=0.5)
plt.show()