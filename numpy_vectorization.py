import time
import numpy as np


#Vector creation: NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((4,))
print(f"np.zeros(4,): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.)
print(f"np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2])
print(f"np.array([5,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([5.,4,3,2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
print()

#Vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#Access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

#Access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#Indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

print()

#Vector slicing operations
a = np.arange(10) #0 - 9
print(f"a = {a}")

c = a[2:7:1]
print(f"a[2:7:1] = {c}")

c = a[2:7:2]
print(f"a[2:7:2] = {c}")

c = a[3:]
print(f"a[3:] = {c}")

c = a[:3]
print(f"a[:3] = {c}")

c = a[:]
print(f"a[:] = {c}")

print()

#Single vector operations
a = np.array([1, 2, 3, 4])
print(f"a: {a}")

#Negate elements of a
b = -a
print(f"b = -a: {b}")

#Sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a): {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2: {b}")

#Vector element-wise operations
#Binary operators (element wise - item after item)
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you will see is:")
    print(e)

#Scalar vector operations
a = np.array([1, 2, 3, 4])
b = 5 * a
print(f"b = 5 * a: {b}")

#Vector dot product (important function - widely used)!
#Dimensions of the two vectors must be the same
def my_dot(a, b):
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b): {my_dot(a, b)}") #Output is scalar value

#Vector dot product using np.dot function
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape}")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape}")

#Vectors vs loops (need for speed)
np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(f"np.dot(a, b) = {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms")

tic = time.time()
c = my_dot(a, b)
toc = time.time()
print(f"my_dot(a, b) = {c:.4f}")
print(f"Loop version duration: {1000*(toc-tic):.4f} ms")

#Remove these big arrays from memory
del(a)
del(b)

print()

X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

print()

#Matrices
#Matrix creation
a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

print()

a = np.array([[5], [4], [3]])
print(f"a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f"a shape = {a.shape}, np.array: a = {a}")

print()

#Vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)
print(f"a.shape: {a.shape}, \na= {a}")

#Access an element
print(f"\na[2,0].shape: {a[2, 0].shape}, a[2,0] = {a[2, 0]}, type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#Access a row
print(f"a[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}")

print()

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ", a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ", a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ", a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ", a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ", a[1].shape   =", a[1].shape, "a 1-D array")