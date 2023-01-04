import numpy as np
import matplotlib.pyplot as plt
import torch
from multiply_by_H_band import multiply_by_H
import numpy as np
from scipy.io import savemat

class MultipleInteractingKF:
    def __init__(self, x, P, Q, F, H, R, b, D, ind, ind_y, nb, proj):
        self.x = x
        self.P = P  # state variance
        self.Q = Q  # process noise variance
        self.F = F  # state transition matrix
        self.H = H  # measurement function
        self.R = R  # measurement noise variance
        self.b = b
        self.D = D  # D matrix for QC code
        self.ind = ind
        self.ind_y = ind_y
        self.nb = nb
        self.proj = proj

    def predict(self):
        x_new_old = torch.zeros(self.x.size())
        P_new_old = torch.zeros(self.Q.size())
        for i in range(len(self.ind)):
            ind_band = self.ind[i].numpy().astype(int)
            x_new_old[ind_band, :] = torch.mm(self.F[ind_band, :][:, ind_band], self.x[ind_band, :])
            P_temp = torch.mm(torch.mm(self.F[ind_band, :][:, ind_band], self.P[ind_band, :][:, ind_band]),
                              self.F[ind_band, :][:, ind_band].T)
            # P_new_old[ind_band, ind_band] = torch.diag(P_temp) + self.Q[ind_band, ind_band]
            for k in range(len(ind_band)):
                P_new_old[ind_band, ind_band[k]:ind_band[k] + 1] = P_temp[:, k:k+1] + self.Q[ind_band, ind_band[k]:ind_band[k] + 1]
        # P_new_old = torch.diag(torch.diag(P_new_old))
        self.x = x_new_old
        self.P = P_new_old
        return x_new_old, P_new_old

    def update(self, y):
        x_new_old = self.x
        P_new_old = self.P
        x_new_new = torch.zeros(self.x.size())
        P_new_new = torch.zeros(self.Q.size())
        # perform operations involving H according to which implementation is being used:
        if isinstance(self.H, multiply_by_H):
            for i in range(len(self.ind)):
                ind_band = self.ind[i].numpy().astype(int)
                ind_band_y = self.ind_y[i].numpy().astype(int)
                z = torch.mm(torch.diag(self.D[ind_band_y, ind_band_y]), y[ind_band_y, :])\
                    - torch.mm(torch.diag(self.D[ind_band_y, ind_band_y]), self.H.H_left(x_new_old[ind_band, :])) \
                    - torch.mm(torch.diag(self.D[ind_band_y, ind_band_y]),
                               torch.ones(y[ind_band_y, :].size()) * self.b)
                S = torch.mm(self.H.Ht_right(torch.mm(torch.diag(self.D[ind_band_y, ind_band_y]),
                                                      self.H.H_left(P_new_old[ind_band, :][:, ind_band]))),
                             torch.diag(self.D[ind_band_y, ind_band_y]).T) \
                    + torch.mm(torch.mm(torch.diag(self.D[ind_band_y, ind_band_y]), torch.diag(self.R[ind_band_y, ind_band_y])),
                               torch.diag(self.D[ind_band_y, ind_band_y]).T)
                sigmayx = torch.mm(self.H.Ht_right(P_new_old[ind_band, :][:, ind_band]), torch.diag(self.D[ind_band_y, ind_band_y]).T)
                k = torch.mm(sigmayx, torch.inverse(S))
                x_new_new[ind_band, :] = x_new_old[ind_band, :] + torch.mm(k, z)
                sigmaS = torch.mm(k, torch.mm(S, k.T))
                # P_new_new[ind_band, ind_band] = P_new_old[ind_band, ind_band] - torch.diag(sigmaS)
                for k in range(len(ind_band)):
                    P_new_new[ind_band, ind_band[k]:ind_band[k]+1] = P_new_old[ind_band, ind_band[k]:ind_band[k]+1] - sigmaS[:, k:k+1]
        else:
            for i in range(len(self.ind)):
                ind_band = self.ind[i].numpy().astype(int)
                z = torch.mm(torch.diag(self.D[ind_band, ind_band]), y[ind_band, :]) \
                    - torch.mm(torch.diag(self.D[ind_band, ind_band]), torch.mm(torch.diag(self.H[ind_band, ind_band]), x_new_old[ind_band, :]))
                S = torch.mm(torch.mm(torch.mm(torch.diag(self.D[ind_band, ind_band]),
                                               torch.mm(torch.diag(self.H[ind_band, ind_band]),
                                                        P_new_old[ind_band, :][:, ind_band])),
                             torch.diag(self.H[ind_band, ind_band]).T), torch.diag(self.D[ind_band, ind_band]).T) \
                    + torch.mm(torch.mm(torch.diag(self.D[ind_band, ind_band]), torch.diag(self.R[ind_band, ind_band])),
                               torch.diag(self.D[ind_band, ind_band]).T)
                k = torch.mm(torch.mm(torch.mm(P_new_old[ind_band, :][:, ind_band],
                                               torch.diag(self.H[ind_band, ind_band]).T),
                                      torch.diag(self.D[ind_band, ind_band]).T), torch.inverse(S))
                x_new_new[ind_band, :] = x_new_old[ind_band, :] + torch.mm(k, z)
                sigmaS = torch.mm(k, torch.mm(S, k.T))
                # P_new_new[ind_band, ind_band] = P_new_old[ind_band, ind_band] - torch.diag(sigmaS)
                for k in range(len(ind_band)):
                    P_new_new[ind_band, ind_band[k]:ind_band[k]+1] = P_new_old[ind_band, ind_band[k]:ind_band[k]+1] - sigmaS[:, k:k+1]
        # P_new_new = torch.diag(torch.diag(P_new_new))
        if self.proj.flag == 1:
            x_new_new = self.proj.projection_func(x_new_new)
        self.x = x_new_new
        self.P = P_new_new
        return x_new_new, P_new_new



