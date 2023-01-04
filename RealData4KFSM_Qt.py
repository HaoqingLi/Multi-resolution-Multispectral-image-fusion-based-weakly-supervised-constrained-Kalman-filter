import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import scipy.io as io
from scipy.io import savemat
import numpy as np
from KFSMFunction import Qest as Qest
from multiply_by_H_band import multiply_by_H
from MIKF_1st import MultipleInteractingKF as MultipleInteractingKF_1st
from MISM_1st import MultipleInteractingSmoother as MultipleInteractingSmoother_1st
from MIKF_2nd import MultipleInteractingKF as MultipleInteractingKF_2nd
from MISM_2nd import MultipleInteractingSmoother as MultipleInteractingSmoother_2nd
from MIKF_3rd import MultipleInteractingKF as MultipleInteractingKF_3rd
from MISM_3rd import MultipleInteractingSmoother as MultipleInteractingSmoother_3rd
from projection import projection
from KFSMFunction import Permutation_Landsat as Permutation_Landsat
from KFSMFunction import Permutation_Modis as Permutation_Modis
from ConfigureFile import *
from PlotOverTime import ReconstructionPlot

datafile = io.loadmat(Originaldatafname)  # key is 'data_temp'
y_LandSat = torch.as_tensor(datafile['observation_LandSat']).float()
y_Modis = torch.as_tensor(datafile['observation_Modis']).float()
order = torch.as_tensor(np.double(datafile['order'])).float()
Init_Modis = int(np.double(datafile['Init_Modis']))
Init_LandSat = int(np.double(datafile['Init_LandSat']))
ind_matrix = torch.as_tensor(datafile['ind_matrix'].astype(int))
norm_factor = torch.as_tensor(datafile["norm_factor"]).float()
HistoricalData = torch.as_tensor(datafile['observation_landsat_all']).float()


## Preprocessing for observations
# Pixel positivity check
HistoricalData[HistoricalData < epsilon] = epsilon
y_LandSat[y_LandSat < epsilon] = epsilon
# Observation Permutation
y_Modis = Permutation_Modis(y_Modis, nb)
PermutationM = Permutation_Landsat(y_LandSat, y_Modis, nb, scale_L)
y_LandSat = torch.mm(PermutationM, y_LandSat)
HistoricalData = torch.mm(PermutationM, HistoricalData)


# Kalman filter Parameters
F = torch.eye(y_LandSat.size(0))
R_Modis = (1e-2)**2 * torch.eye(y_Modis.size(0))
R_LandSat = (1e-5)**2 * torch.eye(y_LandSat.size(0))
Q = torch.eye(y_LandSat.size(0))
D = torch.eye(y_LandSat.size(0))
H_M = torch.reshape(norm_factor/((scale_multi_factor*scale_L)**2)*torch.ones(scale_multi_factor*scale_L, scale_multi_factor*scale_L).float(), [(scale_multi_factor*scale_L)**2, 1])
H_Modis = multiply_by_H(torch.reshape(H_M[:, 0], (scale_L*scale_multi_factor, scale_L*scale_multi_factor)), scale_L, [scale_L, scale_L, nb])
H_Landsat = torch.eye(y_LandSat.size(0))
b = 0
x_old_old = y_LandSat[:, 0:1]
proj_flag = 1       # Flag for projection
proj_min = epsilon  #minimum value for projection
proj_max = 0.5      #maximum value for projection
proj = projection(proj_min, proj_max, proj_flag)  #Projection


#index for split Kalman filter, both Landsat and MODIS observations
group_matrix = torch.zeros(scale_L**2*nb, nl*ns)
group_matrix_y = torch.zeros(nb, nl*ns)
for m in range(nb):
    for i in range(nl*ns):
        group_matrix[:, i] = torch.linspace(i*scale_L**2*nb, (i+1)*scale_L**2*nb-1, scale_L**2*nb)
        group_matrix_y[m, i] = int(nb*i+m)
ind_group = tuple(group_matrix.T)
ind_group_y = tuple(group_matrix_y.T)

# Covariance Matrix
sigmaP = 1e-5
if ModelOption == 'Diagonal':
    # Diagonal Covariance Matrix
    P = sigmaP ** 2 * torch.eye(y_LandSat.size(0))
elif ModelOption == 'BlockDiagonal':
    # # Block Diagonal Covariance Matrix
    ratio = 2
    indnb = torch.linspace(0, nb - 1, nb).numpy().astype(int)
    matrix = torch.ones(nb, nb)
    matrix[indnb, indnb] = ratio * torch.ones(1, nb)
    Ptemp = matrix/ratio
    for i in range(nl*ns*scale_L**2-1):
        matrix = torch.ones(nb, nb)
        matrix[indnb, indnb] = ratio * torch.ones(1, nb)
        Ptemp = torch.block_diag(Ptemp, matrix/ratio)
    P = sigmaP**2*Ptemp
elif ModelOption == 'Full':
    # #Full Covariance Matrix
    ratio = 2
    indnb = torch.linspace(0, nb*scale_L**2 - 1, nb*scale_L**2).numpy().astype(int)
    matrix = torch.ones(nb*scale_L**2, nb*scale_L**2)
    matrix[indnb, indnb] = ratio * torch.ones(1, nb*scale_L**2)
    Ptemp = matrix/ratio
    # Ptemp = torch.ones(nb*scale_L**2, nb*scale_L**2)
    for i in range(nl*ns-1):
        Ptemp = torch.block_diag(Ptemp, matrix/ratio)
    P = sigmaP**2*Ptemp


## Memory Allocate
N = order.size(0)*order.size(1)
x_kf_est = torch.zeros(y_LandSat.size(0), N)
x_sm_est = torch.zeros(y_LandSat.size(0), N)
x_kf_memory = torch.zeros(y_LandSat.size(0), N)
x_sm_memory = torch.zeros(y_LandSat.size(0), N)
y_LandSat = y_LandSat[:, 1:y_LandSat.size(1)]
q_memory = torch.zeros(y_LandSat.size(0), y_LandSat.size(0), N)
P_est = torch.zeros(P.size(0), P.size(1), N)


## Kalman filter and Smoother Initialization
if ModelOption == 'Diagonal':
    kf = MultipleInteractingKF_3rd(x_old_old, P, Q, F, H_Modis, R_Modis, b, D, ind_group, ind_group_y, nb, proj)
    sm = MultipleInteractingSmoother_3rd(Q, F, ind_group, int(y_LandSat.size(0) / nb), nb, proj)
elif ModelOption == 'BlockDiagonal':
    kf = MultipleInteractingKF_2nd(x_old_old, P, Q, F, H_Modis, R_Modis, b, D, ind_group, ind_group_y, nb, proj)
    sm = MultipleInteractingSmoother_2nd(Q, F, ind_group, int(y_LandSat.size(0) / nb), nb, proj)
elif ModelOption == 'Full':
    kf = MultipleInteractingKF_1st(x_old_old, P, Q, F, H_Modis, R_Modis, b, D, ind_group, ind_group_y, nb, proj)
    sm = MultipleInteractingSmoother_1st(Q, F, ind_group, int(y_LandSat.size(0) / nb), nb, proj)
print('multiply_by_H')


## Simulation begins
idi = 0
for i in range(0, N):
    q = Qest(x_old_old, HistoricalData, Historicaldates, alpha)
    deltat = dates_kf[i]
    q = q*deltat
    Q = q + epsilon * torch.eye(nl * ns * scale_L ** 2 * nb)
    q_memory[:, :, i] = Q
    if order[0, i] >= Init_LandSat:
        kf.H = H_Landsat
        kf.R = R_LandSat
        kf.Q = Q
        kf.b = b
        kf.D = torch.eye(y_LandSat.size(0))
        x_new_old, P_new_old = kf.predict()
        x_new_new, P_new_new = kf.update(torch.unsqueeze(y_LandSat[:, int(order[0, i]) - Init_LandSat], 1))
        kf.P = P_new_new
        print(P_new_new.size())
    else:
        kf.H = H_Modis
        kf.R = R_Modis
        kf.b = b
        kf.Q = Q
        kf.D = torch.eye(y_Modis.size(0))
        x_new_old, P_new_old = kf.predict()
        x_new_new, P_new_new = kf.update(torch.unsqueeze(y_Modis[:, int(order[0, i])-Init_Modis], 1))
    x_kf_est[:, i] = x_new_new.view(-1)
    x_true = x_new_new
    x_kf_memory[:, i] = x_true.view(-1)
    P_est[:, :, idi] = P_new_new
    idi += 1
    x_s = x_new_new
    P_s = P_new_new
    x_old_old = x_new_new
    if i == N-1:
        index = i
        x_sm_est[:, index] = x_new_new.view(-1)
        x_true = x_new_new
        x_sm_memory[:, index] = x_true.view(-1)
        for j in range(1, N):
            Q = q_memory[:, :, index-j]
            sm.Q = Q
            x_n_o, P_n_o = sm.predict(torch.unsqueeze(x_kf_est[:, index-j], 1), P_est[:, :, idi-j-1])
            x_n_n, P_n_n, G = sm.update(torch.unsqueeze(x_kf_est[:, index-j], 1), P_est[:, :, idi-j-1], x_s, P_s)
            x_sm_est[:, index - j] = x_n_n.view(-1)
            x_true = x_n_n
            x_sm_memory[:, index - j] = x_true.view(-1)
            x_s = x_n_n
            P_s = P_n_n
        idi = 0
    if (i % 1 == 0):
        print(i)

x_kf_memory = torch.mm(PermutationM.T, x_kf_memory)
x_sm_memory = torch.mm(PermutationM.T, x_sm_memory)

ReconstructionPlot(x_kf_memory, x_sm_memory)

mdic = {"x_kf_est": x_kf_memory.double().numpy(), "x_sm_est":x_sm_memory.double().numpy()}
filename = 'Landsat_2band_' + ModelOption + '_alpha_' + str(alpha) + '_Qt.mat'
savemat(filename, mdic)
