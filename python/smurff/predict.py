#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
import h5sparse

from .result import Prediction

class Sample:
    def __init__(self, h5_file, name, num_latent):
        self.h5_group = h5_file[name]
        self.no = int(self.h5_group.attrs["number"])
        self.nmodes = h5_file["config/options"].attrs['num_priors']
        self.num_latent = num_latent

    def predStats(self):
        # rmse , auc
        return self.h5_group["predictions"].attrs

    def predAvg(self):
        return self.h5_group["predictions/pred_avg"][()]

    def predVar(self):
        return self.h5_group["predictions/pred_var"][()]

    def lookup_mode(self, templ, mode):
        return self.h5_group[templ % mode][()]

    def lookup_modes(self, templ):
        return [ self.lookup_mode(templ, i) for i in range(self.nmodes) ]

    def latents(self):
        return self.lookup_modes("latents/latents_%d")

    def betas(self):
        return self.lookup_modes("link_matrices/link_matrix_%d")
        
    def mus(self):
        return self.lookup_modes("link_matrices/mu_%d")

    def beta_shape(self):
        return [b.shape[0] for b in self.betas()]

    def postMuLambda(self, mode):
        mu = self.lookup_mode("latents/post_mu_%d", mode)
        Lambda = self.lookup_mode("latents/post_lambda_%d", mode)
        assert Lambda.shape[1] == self.num_latent * self.num_latent
        Lambda = Lambda.reshape(Lambda.shape[0], self.num_latent, self.num_latent)

        return mu, Lambda

    def predict(self, params):
        """
        Generate predictions from this `Sample` based on `params`. Parameters
        specify coordinates of sideinfo/features for each dimension.
        See :class:`PredictSession` for clarification.
        """

        if isinstance(params, sp.spmatrix):
            m = params.tocoo()
            p = [ self.predict(x,y) for x,y in zip(m.row, m.col) ]
            return sp.coo_matrix((m.row, m.col), p)

        # else params should be a tuple of len(nmodes)
        assert len(params) == self.nmodes, \
            "You should provide as many parameters as dimensions of the train matrix/tensor"

        # for one prediction: einsum(U[:,coords[0]], [0], U[:,coords[1]], [0], ...)
        # for all predictions: einsum(U[0], [0, 0], U[1], [0, 1], U[2], [0, 2], ...)

        operands = []
        for U, mu, beta, c, m in zip(self.latents(), self.mus(), self.betas(), params, range(self.nmodes)):
            if c is Ellipsis:
                # predict all in this dimension
                operands += [U, [m+1, 0]]
            elif isinstance(c, (int, np.integer)):
                # if a single coord was specified for this dimension,
                # we predict for this coord
                operands += [U[c:c+1, :], [m+1, 0]]
            elif isinstance(c, (range,slice)):
                # if a range/slice was specified for this dimension,
                # we predict for this range/slice
                operands += [U[c, :], [m+1,0]]
            elif isinstance(c, (np.ndarray, sp.spmatrix)):
                if len(c.shape) == 1:
                    assert c.shape[0] == beta.shape[0], \
                        f"Incorrect side-info dims, should be {beta.shape[0]}, got {c.shape}"
                    c = c[np.newaxis, :]
                else:
                    assert len(c.shape) == 2 and c.shape[1] == beta.shape[0], \
                        f"Incorrect side-info dims, should be N x {beta.shape[0]}, got {c.shape}"

                # compute latent vector from side_info using dot
                uhat = c.dot(beta)
                mu = np.squeeze(mu)
                operands += [uhat + mu, [m+1, 0]]
            else:
                raise ValueError("Unknown parameter to predict: " + str(c) + "( of type " + str(type(c)) + ")")

        return np.einsum(*operands)


class PredictSession:
    """TrainSession for making predictions using a model generated using a :class:`TrainSession`.

    A :class:`PredictSession` can be made directly from a :class:`TrainSession`

    >>> predict_session  = train_session.makePredictSession()

    or from a saved hdf5 file

    >>> predict_session = PredictSession("saved_trainsession.hdf5")

    """
    def __init__(self, h5_fname):
        """Creates a :class:`PredictSession` from a given HDF5 file
 
        Parameters
        ----------
        h5_fname : string
           Name of the HDF5 file.
 
        """
        self.h5_file = h5sparse.File(h5_fname, 'r')
        self.options = self.h5_file["config/options"].attrs
        self.nmodes = int(self.options['num_priors'])
        self.num_latent = int(self.options["num_latent"])
        self.data_shape = self.h5_file["config/train/data"].shape

        self.samples = []
        for name in self.h5_file.keys():
            if not name.startswith("sample_"):
                continue

            sample = Sample(self.h5_file, name, self.num_latent)
            self.samples.append(sample)
            self.beta_shape = sample.beta_shape()

        if len(self.samples) == 0:
            raise ValueError("No samples found in " + h5_fname)

        self.samples.sort(key=lambda x: x.no)
        self.num_samples = len(self.samples)

    def lastSample(self):
        return self.samples[-1]

    def postMuLambda(self, mode):
        return self.lastSample().postMuLambda(mode)

    def predictionsYTest(self):
        return self.lastSample().predAvg(), self.lastSample().predVar()

    def statsYTest(self):
        return self.lastSample().predStats()

    def predict(self, operands, samples = None):
        """
        Generate predictions on `operands`. Parameters
        specify coordinates of sideinfo/features for each dimension.

        Parameters
        ----------
        operands: tuple or scipy.sparse matrix

            When operands ia a tuple, it must be combination of coordindates
            in the matrix/tensor and/or features you want to use to make
            predictions. `len(coords)` should be equal to number of
            dimensions in the sample.

            Each element `coords` can be a:
              * :type:`int`: a single element in this dimension is selected. For example, a
                single row or column in a matrix.
              * :class:`slice`: a slice is selected in this dimension. For example, a number of
                rows or columns in a matrix.
              * :type:`Ellipsis`: all elements in this dimension are selected. For example, all
                rows or columns in a matrix.
              * :class:`numpy.ndarray`: 2D numpy array used as dense sideinfo. Each row
                vector is used as side-info.
              * :class:`scipy.sparse.spmatrix`: sparse matrix used as sideinfo. Each row
                vector is used as side-info.

            When operands is a scipy sparse matrix, predictions are made for all non-zero elements 
            in the matrix.

        samples: range or None
            Range of samples to use for prediction, or None for all samples

        Returns
        -------
        numpy.ndarray or list of scipy.sparse matrices
            A :class:`numpy.ndarray` of shape `[ N x T1 x T2 x ... ]` where
            N is the number of samples in this `PredictSession` and `T1 x T2 x ...` 
            has the same numer of dimensions as the train data.

            A scipy.sparse matrix for each sample, where the non-zeros in 
            each matrix are identical to those in the input `operands` parameter.
        """        

        if samples is None:
            samples = self.samples
        else:
            samples = self.samples[samples]

        return np.stack([sample.predict(operands) for sample in samples])

    def __str__(self):
        dat = (len(self.samples), self.data_shape, self.beta_shape, self.num_latent)
        return "PredictSession with %d samples\n  Data shape = %s\n  Beta shape = %s\n  Num latent = %d" % dat
