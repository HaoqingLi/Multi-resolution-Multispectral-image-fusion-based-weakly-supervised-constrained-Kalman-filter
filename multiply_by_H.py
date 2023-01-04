#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:30:23 2020
@author: ricardo
"""
from scipy.ndimage import convolve
from scipy import signal
import numpy as np
import torch
from Decimatefun import Decimatefun
import scipy


class multiply_by_H:
    """Multiply X by H (a bandwise convolution and spatial decimation)
    h_mask : 2D array with the convolution mask
    d : decimation fraction
    dims_high : dimensions (rows,cols,bands) of the HR image
    ==> when the mask or decimation is different for each band of the image,
    the variables are to be supplied as tupes, containing the data for
    each band <=="""

    def __init__(self, h_mask, d, dims_high):
        self.h_mask = h_mask  # convolution mask
        self.d = d  # decimation factor
        self.dims_high = dims_high  # (nl,nc,L), dimensions of the HR image
        # self.dims_low = dims_low  # same as above but for the LR image

        # if the filters/decimation/sizes are tuples, treat each band independently
        if isinstance(h_mask, tuple):
            self.num_bands = len(dims_high)  # get number of bands, before was =dims_high[1][2]
        else:
            self.num_bands = dims_high[2]  # the number of bands must be the same for all elements
            # extend objects to tuples
            self.h_mask = (self.h_mask,) * self.num_bands
            self.d = (self.d,) * self.num_bands
            self.dims_high = (self.dims_high,) * self.num_bands

    def H_left(self, X):
        """" Computes H*X, X=[x1,x2,...,xK], where each xi is
        a lexicographically ordered (vectorized) nr*nc*L image """
        # if input is a tensor, output a tensor too:
        if torch.is_tensor(X):
            flag_tensorv = True
            X = X.numpy()
        else:
            flag_tensorv = False

        if X.ndim == 1:  # if data is only a vector, expand into a something-by-1 matrix
            X = np.expand_dims(X, axis=1)

        for i in range(0, X.shape[1]):

            # reshape the data, from X to a list of images
            Xi = []
            dims_past = 0  # store the number of elements we already got from the tensor
            for j in range(0, self.num_bands):
                nr_j = self.dims_high[j][0]  # number of rows for band j
                nc_j = self.dims_high[j][1]  # number of columns for band j
                tmp_Xi_band_j = X[dims_past:(dims_past + nr_j * nc_j), i]
                dims_past = dims_past + nr_j * nc_j
                Xi.append(np.reshape(tmp_Xi_band_j, (nr_j, nc_j)))  # for each band reshape to (nr_j,nc_j)

                yij = convolve(Xi[j], self.h_mask[j])  # convolution (average the pixels) for band j
                yij = Decimatefun(yij, self.d[j])  # yij = yij[0::self.d,0::self.d]          # decimation
                if j == 0:
                    Bi = np.reshape(yij, (
                    int(nr_j * nc_j / self.d[j] ** 2), 1))  # np.expand_dims(Bi.flatten(order='F'), axis=1)
                else:
                    yij = np.reshape(yij, (
                    int(nr_j * nc_j / self.d[j] ** 2), 1))  # np.expand_dims(Bi.flatten(order='F'), axis=1)
                    Bi = np.concatenate((Bi, yij), axis=0)

            if i == 0:
                # A = np.reshape(Bi, (int(X[:, i:i+1].shape[0]/self.d**2), 1))#np.expand_dims(Bi.flatten(order='F'), axis=1)
                A = Bi;
            else:
                # temp = np.reshape(Bi, (int(X[:, i:i+1].shape[0]/self.d**2), 1))#np.expand_dims(Bi.flatten(order='F'), axis=1)
                # A = np.concatenate((A, temp), axis=1)
                A = np.concatenate((A, Bi), axis=1)

        # if necessary, convert to tensor
        if flag_tensorv:
            A = torch.from_numpy(A)
        return A

    def H_right(self):
        return 'hello world'

    def Ht_left(self):
        return 'hello world'

    def Ht_right(self, X):
        """ Computes X*H' = (H*X')' """
        X = X.T
        A = self.H_left(X)
        A = A.T
        return A





