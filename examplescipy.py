import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr
A = csc_array([[1.,0.],[1.,1.],[0.,1.]],dtype=float)
b = np.array([0.,0.,0.],dtype=float)
x,istop,itn,normr = lsqr(A, b)[:4]

b1 = np.array([1.,0.,-1.],dtype=float)
x,istop,itn, r1norm = lsqr(A,b)[:4]
print(istop)
