import torch
import scipy.io as io
import numpy as np
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


#Parameter Setting


Originaldatafname = './KFSMDataset.mat'  #Original data set
nl = 9                            # image length
ns = 9                            # image width
nb = 2                            # bands
scale = 9                         # scale between Landsat and MODIS images
scale_multi_factor = 1
epsilon = 1e-8
ModelOption = 'Diagonal'
                                 #Diagonal
                                 #BlockDiagonal
                                 #Full
alpha = 0.1                      # Weights for Q matrix Estimation
## Dates for states and histotical data
Historicaldates = [16, 16, 32, 64]
dates_kf = [32, 0, 6, 5, 5, 7, 6, 2, 5, 5, 7, 4, 5, 7, 6, 5, 5, 0]

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
PermutationM = Permutation_Landsat(y_LandSat, y_Modis, nb, scale)
y_LandSat = torch.mm(PermutationM, y_LandSat)
HistoricalData = torch.mm(PermutationM, HistoricalData)


# Kalman filter Parameters
F = torch.eye(y_LandSat.size(0))
R_Modis = (1e-2)**2 * torch.eye(y_Modis.size(0))
R_LandSat = (1e-5)**2 * torch.eye(y_LandSat.size(0))
Q = torch.eye(y_LandSat.size(0))
D = torch.eye(y_LandSat.size(0))
H_M = torch.reshape(norm_factor/((scale_multi_factor*scale)**2)*torch.ones(scale_multi_factor*scale, scale_multi_factor*scale).float(), [(scale_multi_factor*scale)**2, 1])
H_Modis = multiply_by_H(torch.reshape(H_M[:, 0], (scale*scale_multi_factor, scale*scale_multi_factor)), scale, [scale, scale, nb])
H_Landsat = torch.eye(y_LandSat.size(0))
b = 0
x_old_old = y_LandSat[:, 0:1]
proj_flag = 1       # Flag for projection
proj_min = epsilon  #minimum value for projection
proj_max = 0.5      #maximum value for projection
proj = projection(proj_min, proj_max, proj_flag)  #Projection


#index for split Kalman filter, both Landsat and MODIS observations
group_matrix = torch.zeros(scale**2*nb, nl*ns)
group_matrix_y = torch.zeros(nb, nl*ns)
for m in range(nb):
    for i in range(nl*ns):
        group_matrix[:, i] = torch.linspace(i*scale**2*nb, (i+1)*scale**2*nb-1, scale**2*nb)
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
    for i in range(nl*ns*scale**2-1):
        matrix = torch.ones(nb, nb)
        matrix[indnb, indnb] = ratio * torch.ones(1, nb)
        Ptemp = torch.block_diag(Ptemp, matrix/ratio)
    P = sigmaP**2*Ptemp
elif ModelOption == 'Full':
    # #Full Covariance Matrix
    ratio = 2
    indnb = torch.linspace(0, nb*scale**2 - 1, nb*scale**2).numpy().astype(int)
    matrix = torch.ones(nb*scale**2, nb*scale**2)
    matrix[indnb, indnb] = ratio * torch.ones(1, nb*scale**2)
    Ptemp = matrix/ratio
    # Ptemp = torch.ones(nb*scale**2, nb*scale**2)
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
