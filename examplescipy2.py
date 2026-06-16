import numpy as np
from scipy.optimize import nnls

A = np.array([[1,0],[1,0],[0,1]])
b = np.array([2,1,1])
k = nnls(A,b)
print(k)

b = np.array([-1,-1,-1])
z = nnls(A,b)
print(z)

