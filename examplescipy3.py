import numpy as np
from scipy import sparse, linalg,stats
from scipy.sparse.linalg import svds, aslinearoperator,LinearOperator

# Construct a dense matrix A from singular values and vectors
rng = np.random.default_rng()
orthogonal = stats.ortho_group.rvs(10,random_state=rng)
s = [1e-3, 1 , 2 , 3 , 4]# non-zero singular values
u = orthogonal[:,:5] # left singular vectors
vT = orthogonal[:,5:].T # right singular vectors
A = u @ np.diag(s) @ vT

#with only four singular values/vectors, the SVD approximates the original matrix
u4, s4, vT4 = svds(A, k = 4)
A4 = u4 @ np.diag(s4) @ vT4
y1 = np.allclose(A4,A,atol=1e-3)
print(y1)

#with all five non-zero singular values/vectors, 
# we can reproduce the original matrix more accurately
u5,s5,vT5 = svds(A, k = 5)
A5 = u5 @ np.diag(s5) @ vT5
y2 = np.allclose(A5,A)
print(y2)

#the singular values match the expected singular values
y3 = np.allclose(s5,s)
print(y3)


#since the singular are not close to each other in this example,
#every singular vector matches as expected up to a difference in sign.
y4 = (np.allclose(np.abs(u5),np.abs(u)) and np.allclose(np.abs(vT5),np.abs(vT)))
print(y4)

# The singular vectors are als orthogonal
y5 = (np.allclose(u5.T @ u5, np.eye(5))) and np.allclose(vT5 @ vT5.T,np.eye(5))
print(y5)


# If there are (nearly) multiple singular values, 
# the corresponding individual singular vectors may be unstable,
# but the whole invariant subspace containing all such singular
# vectors is computed accurately as can be measured by angles
# between subspaces via 'subspace_angles'.

rng = np.random.default_rng()
s = [1,1 + 1e-6] # non-zero singular values
u, _ = np.linalg.qr(rng.standard_normal((99,2)))
v, _ = np.linalg.qr(rng.standard_normal((99,2)))
vT = v.T
A = u @ np.diag(s) @ vT
A = A.astype(np.float32)
u2,s2,vT2 = svds(A, k = 2,rng = rng)
y6 = np.allclose(s2,s)
print(y6)

#the angles between the individual exact and computed singular 
#vectors may not be so small.To check use
y7 = (linalg.subspace_angles(u2[:,:1],u[:,:1]) + linalg.subspace_angles(u2[:,1:], u[:,1:]))
print(y7)

y8 = (linalg.subspace_angles(vT2[:1,:].T,vT[:1, :].T) + linalg.subspace_angles(vT2[1:,:].T,vT[1:,:].T))
print(y8)

# As opposed to the angles between the 2-dimensional invariant
# subspaces that these vectors span, which are small for rights singular vectors

y9 = linalg.subspace_angles(u2,u).sum() < 1e-6
print(y9)

# as well as for left singular vectors
y10 = linalg.subspace_angles(vT2.T,vT.T).sum() < 1e-6
print(y10)

#The next example follows that of 'sklearn.decomposotion.TruncatedSVD'
rng = np.random.default_rng()
X_dense = rng.random(size=(100,100))
X_dense[:, 2 * np.arange(50)] = 0
X = sparse.csr_array(X_dense)
_,singular_values, _ = svds(X, k = 5,rng = rng)
print(singular_values)

#The function can be called without the transpose of the input matrix ever explicity constructed

rng = np.random.default_rng()
G = sparse.random_array((8,9), density = 0.5,rng=rng)
Glo = aslinearoperator(G)
_, singular_values_svds, _ = svds(Glo,k = 5,rng = rng)
_,singular_values_svd, _ = linalg.svd(G.toarray())
y11 = np.allclose(singular_values_svds, singular_values_svd[-4::-1])
print(y11)

# The most memory efficient scenario is where neither the original matrix,
# nor its transpose, is explicitly constructed. our example computes the
# smallest singular values and vectors of 'LinearOperator' constructed
# from the numpy function 'np.diff' used column-wise to be consistent 
# with 'LinearOperator' operating on columns

diff0 = lambda a: np.diff(a,axis=0)

#let us create the matrix from 'diff0' to be used for validation only.
n = 5 # The dimension of the space
M_from_diff0 = diff0(np.eye(n))
print(M_from_diff0.astype(int))

#The matrix 'M_from_diff0' is bi_diagonal and could be alternatively created directly by
M = - np.eye(n - 1,n,dtype=int)
y12 = np.fill_diagonal(M[:,1:],1)
y13 = np.allclose(M,M_from_diff0)

#its transpose:
print(M.T)

# the graph Laplacian, while the actually used in 'svds' smaller size 4 * 4 normal matrix M @ M.T

print(M.T @ M)

# The 'LinearOperator' setup needs the options 'rmatvec' and 'rmatmat' of
# multiplication by the matrix transpose M.T, but we want to be matrix-free
# to save memory, so knowing how M.T looks like,
# we manually construct the following function to be used in
# rmatmat = diff0t

def diff0t(a):
    if a.ndim == 1:
        a = a[:,np.newaxis] # Turn 1D into 2D array
    d = np.zeros((a.shape[0] + 1, a.shape[1]),dtype = a.dtype)
    d[0,:] = -a[0,:]
    d[1:-1 :] = a[0:-1, :] - a[1:,:]
    d[-1, :] = a[-1, :]
    return d

# we check that our function 'diff0t' for the matrix transpose is valid
y14 = np.allclose(M.T ,diff0t(np.eye(n-1)))
print(y14)

#Now we setup our matrix-free 'LinearOperator' called 'diff0_func_aslo' and for validation
#the matrix-based 'diff0_matrix_aslo'

def diff0_func_aslo_def(n):
    return LinearOperator(matvec=diff0,
                          matmat=diff0,
                          rmatvec=diff0t,
                          rmatmat=diff0t,
                          shape=(n-1,n))
diff0_func_aslo = diff0_func_aslo_def(n)
diff0_matrix_aslo = aslinearoperator(M_from_diff0)

#And validate both the matrix and its transpose in 'LinearOperator'

y15 = np.allclose(diff0_func_aslo(np.eye(n)),diff0_matrix_aslo(np.eye(n)))
print(y15)

y16 = np.allclose(diff0_func_aslo.T(np.eye(n-1)),diff0_matrix_aslo.T(np.eye(n-1)))
print(y16)

#Having the 'linearOperator' setup validated, we run the solver
n = 100
diff0_func_aslo = diff0_func_aslo_def(n)
u, s, vT = svds(diff0_func_aslo, k = 3, which='SM')

se = 2. * np.sin(np.pi * np.arrange(1,4) / (2. * n))
ue = np.sqrt(2 / n) * np.sin(np.pi * np.outer(np.arange(1,n), np.arange(1,4)) / n)
y17 = np.allclose(s,se,atol=1e-3)
print(y17)
y18 = np.allclose(np.abs(u),np.abs(ue),atol=1e-6)
print(y18)


