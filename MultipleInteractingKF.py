import numpy as np
import matplotlib.pyplot as plt
import torch
from multiply_by_H import multiply_by_H
import numpy as np
from scipy.io import savemat

class MultipleInteractingKF:
    def __init__(self, x, P, Q, F, H, R, b, D, ind, xlen, nb, PT_matrix):
        self.x = x
        self.P = P  # state variance
        self.Q = Q  # process noise variance
        self.F = F  # state transition matrix
        self.H = H  # measurement function
        self.R = R  # measurement noise variance
        self.b = b
        self.D = D  # D matrix for QC code
        self.ind = ind
        self.length = xlen
        self.nb = nb
        self.PT = PT_matrix

    def predict(self):
        x_new_old = torch.zeros(self.x.size())
        P_new_old = torch.zeros(self.Q.size())
        for i in range(len(self.ind)):
            ind_band = self.ind[i].numpy().astype(int)
            x_new_old[ind_band, :] = torch.mm(self.F[ind_band, :], self.x)
            P_temp = torch.mm(torch.mm(self.F[ind_band, :], self.P), self.F[ind_band, :].T)
            for iind in range(len(ind_band)):
                P_new_old[ind_band, ind_band[iind]:ind_band[iind]+1] = P_temp[:, iind:iind+1]\
                                                               + self.Q[ind_band, ind_band[iind]:ind_band[iind]+1]
        P_new_old = torch.diag(torch.diag(P_new_old))
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
            z = torch.mm(self.D, y) - torch.mm(self.D, self.H.H_left(torch.mm(self.PT.T, x_new_old))) - torch.mm(self.D, torch.ones(
                y.size()) * self.b)
            S = torch.mm(self.H.Ht_right(torch.mm(self.D, self.H.H_left(torch.mm(torch.mm(self.PT.T, P_new_old), self.PT )))), self.D.T) + torch.mm(
                torch.mm(self.D, self.R), self.D.T)
            # PHmatrix = self.H.Ht_right(torch.mm(P_new_old, self.PT))
            for i in range(len(self.ind)):
                ind_band = self.ind[i].numpy().astype(int)
                sigmayx = torch.mm(self.H.Ht_right(torch.mm(P_new_old[ind_band, :], self.PT)), self.D.T)
                # sigmayx = torch.mm(self.H.Ht_right(P_new_old[ind_band, :]), self.D.T)
                # sigmayx = torch.mm(PHmatrix[ind_band, :], self.D.T)
                k = torch.mm(sigmayx, torch.inverse(S))
                x_new_new[ind_band, :] = x_new_old[ind_band, :] + torch.mm(k, z)
                sigmaS = torch.mm(k, torch.mm(S, k.T))
                for iind in range(len(ind_band)):
                    P_new_new[ind_band, ind_band[iind]:ind_band[iind]+1] = P_new_old[ind_band, ind_band[iind]:ind_band[iind]+1]\
                                                                            - sigmaS[:, iind:iind+1]
        else:
            z = torch.mm(self.D, y) - torch.mm(self.D, torch.mm(self.H, torch.mm(self.PT.T, x_new_old)))
            S = torch.mm(torch.mm(torch.mm(self.D, torch.mm(self.H, torch.mm(torch.mm(self.PT.T, P_new_old), self.PT ))), self.H.T), self.D.T) + torch.mm(
                torch.mm(self.D, self.R), self.D.T)
            for i in range(len(self.ind)):
                ind_band = self.ind[i].numpy().astype(int)
                k = torch.mm(torch.mm(torch.mm(torch.mm(P_new_old[ind_band, :], self.PT), self.H.T), self.D.T), torch.inverse(S))
                x_new_new[ind_band, :] = x_new_old[ind_band, :] + torch.mm(k, z)
                sigmaS = torch.mm(k, torch.mm(S, k.T))
                for iind in range(len(ind_band)):
                    P_new_new[ind_band, ind_band[iind]:ind_band[iind] + 1] = P_new_old[ind_band,
                                                                             ind_band[iind]:ind_band[iind] + 1] \
                                                                             - sigmaS[:, iind:iind + 1]
        P_new_new = torch.diag(torch.diag(P_new_new))
        self.x = x_new_new
        self.P = P_new_new
        return x_new_new, P_new_new



