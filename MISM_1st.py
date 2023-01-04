import torch
import numpy as np

class MultipleInteractingSmoother:
    def __init__(self, Q, F, ind, xlen, nb, proj):
        self.x = torch.zeros(Q.size())
        self.P = torch.zeros(Q.size())
        self.Q = Q  # process noise variance
        self.F = F  # state transition matrix
        self.ind = ind
        self.length = xlen
        self.nb = nb
        self.proj = proj

    def predict(self, x, P):
        x_new_old = torch.zeros(x.size())
        P_new_old = torch.zeros(self.Q.size())
        for i in range(len(self.ind)):
            ind_band = self.ind[i].numpy().astype(int)
            x_new_old[ind_band, :] = torch.mm(self.F[ind_band, :][:, ind_band], x[ind_band, :])
            P_temp = torch.mm(torch.mm(self.F[ind_band, :][:, ind_band], P[ind_band, :][:, ind_band]),
                              self.F[ind_band, :][:, ind_band].T)
            # P_new_old[ind_band, ind_band] = torch.diag(P_temp) + self.Q[ind_band, ind_band]
            for k in range(len(ind_band)):
                P_new_old[ind_band, ind_band[k]:ind_band[k] + 1] = P_temp[:, k:k+1] + self.Q[ind_band, ind_band[k]:ind_band[k] + 1]
        # P_new_old = torch.diag(torch.diag(P_new_old))
        self.x = x_new_old
        self.P = P_new_old
        return x_new_old, P_new_old

    def update(self, x, P, x_s, P_s):
        x_new_old = self.x
        P_new_old = self.P + 1e-40*torch.eye(self.P.size(0))

        x_old_old = torch.zeros(self.x.size())
        P_old_old = torch.zeros(self.Q.size())
        for i in range(len(self.ind)):
            ind_band = self.ind[i].numpy().astype(int)
            G = torch.mm(torch.mm(P[ind_band, :][:, ind_band], self.F[ind_band, :][:, ind_band].T), torch.inverse(P_new_old[ind_band, :][:, ind_band]))
            z = x_s[ind_band, :] - x_new_old[ind_band, :]
            x_old_old[ind_band, :] = x[ind_band, :] + torch.mm(G, z)
            Ptemp = torch.mm(torch.mm(G, (P_s[ind_band, :][:, ind_band] - P_new_old[ind_band, :][:, ind_band])), G.T)
            # print('Psize:' + str(Ptemp.size()))
            # P_old_old[ind_band, ind_band] = P[ind_band, ind_band] - torch.diag(Ptemp)
            for k in range(len(ind_band)):
                P_old_old[ind_band, ind_band[k]:ind_band[k] + 1] = P[ind_band, ind_band[k]:ind_band[k] + 1] + Ptemp[:, k:k+1]
        # P_old_old = torch.diag(torch.diag(P_old_old))
        if self.proj.flag == 1:
            x_old_old = self.proj.projection_func(x_old_old)
        return x_old_old, P_old_old, G
