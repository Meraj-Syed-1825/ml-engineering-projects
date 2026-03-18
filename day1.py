import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("\n✅ All good — ready to go!")


# BLock 1 : Creating Arrays
import numpy as np

# 1D vector
a = np.array([1,2,3])

# 2D matrix
A = np.array([[1,2], [3,4]])

print(a)
print(A)

print("Shape:", A.shape) # rows and colums
print("Type:", A.dtype)

# Block 2
print(np.zeros((3,3))) # zero matrix
print(np.ones((2,4))) # one matrix
print(np.eye(3)) # identity matrix
print(np.linspace(0,1,5)) # 5 even;y spaced points from 0 to 1

A = np.array([[1,2], [3,4]], dtype = float)
print(A.dtype)

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])

print(A + B) # element - wise addition
print(A * B) # element wise multiplication
print (A @ B) # matrix multiplication
print(A.T) # transpose matrix

print(np.linalg.inv(A))
print(np.linalg.det(A))
# if result == -2.0:   # dangerous!
# if np.isclose(result, -2.0):   # correct way,  Always use this way never ==

print(np.linalg.det(A) == -2.0)        # what does this give?
print(np.isclose(np.linalg.det(A), -2.0))  # and this?

A = np.arange(12).reshape (3,4)
print(A)

print(A[0, :])
print(A[:, 1])
print(A[1:3, 0:2])

# mini exercise

u = np.array([0.0, 0.12, 0.31, 0.28, 0.05]) # nodal displacements
k = np.array([210, 185, 195, 200, 178])
F = k * u
print("Forces:", F)
print("Mean Force:", np.mean(F))
print("Max Force:", np.max(F))
print("Element with max Force:", np.argmax(F))

"""
That's Day 1 complete! Here's what you actually learned today:

NumPy arrays, shapes, dtypes
Creating special matrices (zeros, ones, eye, linspace)
Matrix math — especially the difference between * and @
Float precision errors and np.isclose()
Slicing and indexing
Aggregate functions: mean, max, argmax

"""