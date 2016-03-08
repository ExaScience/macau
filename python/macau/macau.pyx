cimport cython                                                                                                         
import numpy as np
cimport numpy as np
import scipy as sp
import timeit

cimport macau

## using cysignals to catch CTRL-C interrupt
include "cysignals/signals.pxi"


cpdef mul_blas(np.ndarray[np.double_t, ndim=2] x, np.ndarray[np.double_t, ndim=2] y):
    if y.shape[0] != y.shape[1]:
        raise ValueError("y must be square matrix.")
    if x.shape[1] != y.shape[0]:
        raise ValueError("num columns in x and y must be the same.")
    hello(&x[0,0], &y[0,0], x.shape[1], x.shape[0])

cpdef py_getx():
    cdef MatrixXd m = getx()
    cdef np.ndarray[np.double_t, ndim=2] A = matview(&m)
    #print(A)
    return A.copy()

cpdef py_eigenQR(np.ndarray[np.double_t, ndim=2] x):
    eigenQR(& x[0,0], x.shape[0], x.shape[1])

cpdef blas_AtA(np.ndarray[np.double_t, ndim=2] X):
    cdef MatrixXd Xeig = Map[MatrixXd](&X[0,0], X.shape[0], X.shape[1])
    cdef np.ndarray[np.double_t, ndim=2] AtA = np.zeros((X.shape[1], X.shape[1]))
    At_mul_A_blas(Xeig, & AtA[0,0])
    return AtA

cdef SparseFeat sparse2SparseBinFeat(X, int row_block_size = 1024, int col_block_size = 1024, bool sort = True):
    X = X.tocoo(copy=False)
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    cdef SparseFeat K
    K = SparseFeat(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0], sort, row_block_size, col_block_size)
    return K

cpdef blockcg(X, np.ndarray[np.double_t, ndim=2] B, double reg, int row_block_size = 1024, int col_block_size = 1024, double tol = 1e-6):
    if not np.isfortran(B):
        raise ValueError("B must have order='F' (fortran)")
    if B.shape[1] != X.shape[1]:
        raise ValueError("B.shape[1] must equal X.shape[1]")
    X = X.tocoo(copy=False)
    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros( (B.shape[0], B.shape[1]), order='F' )
    cdef MatrixXd result_eig = Map[MatrixXd](&result[0,0], result.shape[0], result.shape[1])

    cdef MatrixXd Beig = Map[MatrixXd](&B[0,0], B.shape[0], B.shape[1])
    cdef MatrixXd out  = Map[MatrixXd](&B[0,0], B.shape[0], B.shape[1])
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    cdef SparseFeat K
    print("Sorting sparse [%d x %d] matrix" % (X.shape[0], X.shape[1]))
    K = SparseFeat(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0], True, row_block_size, col_block_size)
    print("Running block-cg")
    cdef double start = timeit.default_timer()
    cdef int niter = solve(result_eig, K, reg, Beig, tol)
    cdef double end = timeit.default_timer()

    cdef np.ndarray[np.double_t] v = np.zeros(2)
    v[0] = niter
    v[1] = end - start
    return v
    #cdef np.ndarray[np.double_t] ivals = X.data.astype(np.double, copy=False)

cdef matview(MatrixXd *A):
    cdef int nrow = A.rows()
    cdef int ncol = A.cols()
    cdef np.double_t[:,:] view = <np.double_t[:nrow:1, :ncol]> A.data()
    return np.asarray(view)

cdef vecview(VectorXd *v):
    cdef int size = v.size()
    cdef np.double_t[:] view = <np.double_t[:size]> v.data()
    return np.asarray(view)

## API functions:
## 1) F'F
## 2) F*X (X is a matrix)
## 3) F'X (X is a matrix)
## 4) solve(A, b, sym_pos=True) where A is posdef

def bpmf(Y,
         Ytest = None,
         num_latent = 10,
         precision  = 1.0, 
         burnin     = 50,
         nsamples   = 400):
    if type(Y) != sp.sparse.coo.coo_matrix:
        raise ValueError("Y must be scipy.sparse.coo.coo_matrix")
    if Ytest != None:
        if type(Ytest) != sp.sparse.coo.coo_matrix:
            raise ValueError("Ytest must be scipy.sparse.coo.coo_matrix")
        if Ytest.shape != Y.shape:
            raise ValueError("Ytest and Y must have the same shape")

    cdef np.ndarray[int] irows = Y.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = Y.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] ivals = Y.data.astype(np.double, copy=False)

    sig_on()
    cdef Macau macau
    macau = Macau(np.int32(num_latent))
    macau.setPrecision(np.float64(precision))
    macau.setRelationData(&irows[0], &icols[0], &ivals[0], irows.shape[0], Y.shape[0], Y.shape[1]);
    macau.setSamples(np.int32(burnin), np.int32(nsamples))

    cdef np.ndarray[int] trows, tcols
    cdef np.ndarray[np.double_t] tvals

    if Ytest != None:
        trows = Ytest.row.astype(np.int32, copy=False)
        tcols = Ytest.col.astype(np.int32, copy=False)
        tvals = Ytest.data.astype(np.double, copy=False)
        macau.setRelationDataTest(&trows[0], &tcols[0], &tvals[0], trows.shape[0], Y.shape[0], Y.shape[1])

    macau.run()
    sig_off()

    #print("rows=%d, cols=%d" % ( macau.prior_u.Lambda.rows(), macau.prior_u.Lambda.cols()))
    #cdef np.ndarray[np.double_t, ndim=2] L = matview(&macau.prior_u.Lambda)
    #cdef np.ndarray[np.double_t] mu = vecview(&macau.prior_u.mu)
    return dict(rmse_test = macau.getRmseTest())

