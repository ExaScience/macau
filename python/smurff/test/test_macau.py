import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import smurff
import itertools
import collections

class TestSmurff(unittest.TestCase):

    # Python 2.7 @unittest.skip fix
    __name__ = "TestSmurff"

    def test_macau(self):
        Ydense  = np.random.rand(10, 20)
        r       = np.random.permutation(10*20)[:40] # 40 random samples from 10*20 matrix
        side1   = Ydense[:,1:2]
        side2   = Ydense[1:2,:].transpose()
        Y       = scipy.sparse.coo_matrix(Ydense) # convert to sparse
        Y       = scipy.sparse.coo_matrix( (Y.data[r], (Y.row[r], Y.col[r])), shape=Y.shape )
        Y, Ytest = smurff.make_train_test(Y, 0.5)

        results = smurff.macau(Y,
                                Ytest=Ytest,
                                side_info=[side1, side2],
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)

        self.assertEqual(Ytest.nnz, len(results.predictions))

    def test_macau_side_bin(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        smurff.macau(X,
                      Ytest=Xt,
                      side_info=[F, None],
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_macau_dense(self):
        Y  = scipy.sparse.rand(15, 10, 0.2)
        Yt = scipy.sparse.rand(15, 10, 0.1)
        F  = np.random.randn(15, 2)
        smurff.macau(Y,
                      Ytest=Yt,
                      side_info=[F, None],
                      num_latent=5,
                      burnin=10,
                      nsamples=5,
                      verbose=False)

    def test_macau_univariate(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = smurff.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = smurff.macau(Y,
                                Ytest=Ytest,
                                side_info=[side1, side2],
                                univariate = True,
                                num_latent=4,
                                verbose=False,
                                burnin=50,
                                nsamples=50)
        self.assertEqual(Ytest.nnz, len(results.predictions))

    def test_macau_tensor(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(3, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        results = smurff.macau(Ytrain = Ytrain,
			 Ytest = Ytest,
			 side_info=[Acoo, None, None],
			 num_latent = 4,
			 verbose = False,
			 burnin = 20,
			 nsamples = 20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

    def test_macau_tensor_univariate(self):
        A = np.random.randn(30, 2)
        B = np.random.randn(4, 2)
        C = np.random.randn(2, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
        Ytrain, Ytest = smurff.make_train_test_df(df, 0.2)

        Acoo = scipy.sparse.coo_matrix(A)

        results = smurff.macau(Ytrain,
                                Ytest=Ytest,
                                side_info=[Acoo, None, None],
                                univariate = True,
                                num_latent=4,
                                verbose=False,
                                burnin=20,
                                nsamples=20)

        self.assertTrue(results.rmse < 0.5,
                        msg="Tensor factorization gave RMSE above 0.5 (%f)." % results.rmse)

if __name__ == '__main__':
    unittest.main()
