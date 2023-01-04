
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np


def var_est(x, dates):
    Q = torch.mm((x[:, 0:1] - x[:, 1:2]), (x[:, 0:1] - x[:, 1:2]).T) / dates
    return Q


def Qest (x, y_LandSat_all, dates_landsat_all, alpha):
    similarity = torch.zeros(1, y_LandSat_all.size(1))
    for i in range(y_LandSat_all.size(1)):
        similarity[0, i] = torch.mm(x.T, y_LandSat_all[:,i:i+1])/np.sqrt(torch.sum(x**2))/np.sqrt(torch.sum(y_LandSat_all[:,i:i+1]**2))
    ind = (similarity[0,:] == torch.max(similarity)).nonzero(as_tuple=True)[0]
    if ind <= y_LandSat_all.size(1) - 2:
        Q = var_est(y_LandSat_all[:, ind:ind+2], dates_landsat_all[ind])
    else:
        Q = var_est(y_LandSat_all[:, y_LandSat_all.size(1) - 2:y_LandSat_all.size(1)], dates_landsat_all[y_LandSat_all.size(1) - 2])
    Qest = (1-alpha)*Q + alpha*torch.diag(torch.diag(Q))
    return Qest


def Permutation_Landsat(y_LandSat, y_Modis, nb, scale):
    PermutationM = torch.zeros(int(y_LandSat.size(0)), int(y_LandSat.size(0)))
    PerM_ind = torch.linspace(0, int(y_LandSat.size(0) / nb) - 1, int(y_LandSat.size(0) / nb))
    PerM_ind = torch.reshape(PerM_ind, (int(np.sqrt(y_LandSat.size(0) / nb)), int(np.sqrt(y_LandSat.size(0) / nb))))
    PerMi_ind = torch.reshape(PerM_ind[0:scale, 0:scale], (scale ** 2, 1))
    for i in range(1, int(y_Modis.size(0) / nb)):
        indx = int(i / np.sqrt(int(y_Modis.size(0) / nb)))
        indy = int(np.mod(i, np.sqrt(int(y_Modis.size(0) / nb))))
        temp = torch.reshape(PerM_ind[indx * scale: (indx + 1) * scale, indy * scale: (indy + 1) * scale],
                             (scale ** 2, 1))
        PerMi_ind = torch.cat((PerMi_ind, temp), axis=0)
    PerM_ind = torch.cat((PerMi_ind, PerMi_ind + int(y_LandSat.size(0) / nb)), axis=0)
    PerM = torch.zeros(PerM_ind.size())
    for i in range(PerM_ind.size(1)):
        Bi_temp = PerM_ind[0: int(PerM_ind.size(0) / nb), i:i + 1]
        for j in range(1, nb):
            Bi_temp = torch.cat(
                (Bi_temp, PerM_ind[int(PerM_ind.size(0) / nb) * j: int(PerM_ind.size(0) / nb) * (j + 1), i:i + 1]),
                axis=1)
        Bi = torch.reshape(Bi_temp, (PerM.size(0), 1))
        PerM[:, i:i + 1] = Bi
    for i in range(PermutationM.size(0)):
        PermutationM[i, int(PerM[i])] = 1
    return PermutationM


def Permutation_Modis(X, band):
    Xtemp = torch.zeros(X.size())
    for i in range(X.size(1)):
        Bi_temp = X[0: int(X.size(0)/band), i:i+1]
        for j in range(1, band):
            Bi_temp = torch.cat((Bi_temp, X[int(X.size(0)/band) * j: int(X.size(0)/band) * (j + 1), i:i+1]), axis=1)
        Bi = torch.reshape(Bi_temp, (Xtemp.size(0), 1))
        Xtemp[:, i:i+1] = Bi
    return Xtemp
