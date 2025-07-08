####################################
# Author: Marek Kowolowski
# Roadmap for my new job next year
# Day: 3.
# Numpy: Arrays, broadcasting
####################################

import numpy as np

# Basic array
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Random numbers
random_arr = np.random.randint(0, 100, size=10) # 10 numbers from 0 to 99
print(random_arr)

# Basic computation
print(f"Sum: {random_arr.sum()}")
print(f"Mean: {random_arr.mean()}")
print(f"Max: {random_arr.max()}")
print(f"Min: {random_arr.min()}")

# Broadcasting - automatic dimension expansion
print("+10 for every item")
print(random_arr + 10)

print("Multiply by 2")
print(random_arr * 2)