
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from scipy.io import savemat
import torch
import numpy as np
from vectorization import vectorization
from multiply_by_H import multiply_by_H
import imReader as imReader
from ConfigureFile import *



H_M = torch.reshape(1 / ((scale_multi_factor * scale_L) ** 2) * torch.ones(scale_multi_factor * scale_L,
                                                                           scale_multi_factor * scale_L).float(),
                    [(scale_multi_factor * scale_L) ** 2, 1])
H_Modis = multiply_by_H(torch.reshape(H_M[:, 0], (scale_L * scale_multi_factor, scale_L * scale_multi_factor)), scale_L,
                        [nl * scale_L, ns * scale_L, nb])

# Modis Image Generation
date_str_list = imReader.get_Moids_list_from_files(dir_modis, root_str_modis)

# storing multiple multi-band images for all dates in date_str_list
modis_imgs = []
for date in date_str_list:
    modis_imgs.append(imReader.ReadMODIS(dir_modis, root_str_modis, date))

modis_size = modis_imgs[0].shape
modis = torch.zeros(modis_imgs.__len__(), modis_size[0], modis_size[1], modis_size[2])
modis_date = torch.zeros(modis_imgs.__len__(), 1, dtype=int)
for i in range(modis_imgs.__len__()):
    for j in range(modis_size[0]):
        modis[i, j, :, :] = torch.as_tensor(modis_imgs[i].img_dict['B' + str(j + 1)])
    modis_date[i, :] = int(modis_imgs[i].date)

# LandSat Image Generation
date_str_list = imReader.get_LandSatdate_list_from_files(dir_landsat, root_str_landsat)
date_str_list = imReader.unique(date_str_list)
landsat_imgs = []
shape = [x * scale_L for x in modis_size]
for date in date_str_list:
    landsat_imgs.append(imReader.ReadLandsat8(dir_landsat, root_str_landsat, date, shape))

landsat_size = landsat_imgs[0].shape
landsat = torch.zeros(landsat_imgs.__len__(), landsat_size[0], landsat_size[1], landsat_size[2])
landsat_date = torch.zeros(landsat_imgs.__len__(), 1, dtype=int)
for i in range(landsat_imgs.__len__()):
    for j in range(landsat_size[0]):
        bandindice = 'B' + str(j + 4)
        landsat[i, j, :, :] = torch.as_tensor(landsat_imgs[i].img_dict[bandindice])
    landsat_date[i, :] = int(landsat_imgs[i].date)


modis_date_chosen_index = [int((modis_date == x).nonzero()[0][0]) for x in modis_date_chosen]
K = len(modis_date_chosen_index)
modis_band_chosen_ind = list(range(len(modis_band_chosen)))
observation_Modis = torch.zeros((x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(modis_band_chosen), K)
for iband in range(len(modis_band_chosen_ind)):
    for i in range(K):
        y_Modis_temp = []
        Image_Modis_temp = []
        Image_Modis = modis[modis_date_chosen_index[i], modis_band_chosen_ind[iband], :, :]
        Y_Modis = Image_Modis[y_lims[0]:y_lims[1] + 1, x_lims[0]:x_lims[1] + 1]
        observation_Modis[iband * Y_Modis.numel(): (iband + 1) * Y_Modis.numel(), i:i + 1] = torch.reshape(Y_Modis, [
            Y_Modis.numel(), 1])


# LandSat Image Chosen
landsat_date_chosen_index = [int((landsat_date == x).nonzero()[0][0]) for x in landsat_date_chosen]
K = len(landsat_date_chosen_index)
landsat_band_chosen_ind = list(range(len(landsat_band_chosen)))
observation_landsat = torch.zeros(
    (x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(landsat_band_chosen) * scale_L ** 2, K)
for iband in range(len(landsat_band_chosen_ind)):
    for i in range(K):
        y_landsat_temp = []
        Image_landsat_temp = []
        Image_landsat = landsat[landsat_date_chosen_index[i], landsat_band_chosen_ind[iband], :, :]
        Y_landsat = Image_landsat[y_lims[0] * scale_L:(y_lims[1] + 1) * scale_L,
                    x_lims[0] * scale_L:(x_lims[1] + 1) * scale_L]
        observation_landsat[iband * Y_landsat.numel(): (iband + 1) * Y_landsat.numel(), i:i + 1] = torch.reshape(
            Y_landsat, [Y_landsat.numel(), 1])

landsat_date_chosen = landsat_date_chosen[1:len(landsat_date_chosen)]

bandsnum = len(modis_band_chosen)
index = torch.unsqueeze(torch.linspace(0, len(observation_landsat) - 1, len(observation_landsat)), 0)
index_matrix = torch.reshape(index, [int(np.sqrt(len(observation_landsat) / bandsnum)),
                                     int(np.sqrt(len(observation_landsat) / bandsnum)), bandsnum])
window_size = 9
group_matrix = torch.zeros(window_size ** 2 * bandsnum, int(len(observation_landsat) / window_size ** 2 / bandsnum))
for i in range(int(len(observation_landsat) / window_size ** 2 / bandsnum)):
    group_matrix[:, i] = index[0, i * window_size ** 2 * bandsnum:(i + 1) * window_size ** 2 * bandsnum]

ind_matrix = torch.zeros(index.numel(), 1)
for i in range(int(np.sqrt(len(observation_landsat) / bandsnum) / window_size)):
    for j in range(int(np.sqrt(len(observation_landsat) / bandsnum) / window_size)):
        ind_matrix[
        int(i * np.sqrt(len(observation_landsat) * bandsnum) * window_size + j * window_size ** 2 * bandsnum):
        int(i * np.sqrt(len(observation_landsat) * bandsnum) * window_size + (j + 1) * window_size ** 2 * bandsnum),
        0:1] \
            = vectorization(
            index_matrix[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size, :])

ind = np.array(modis_date_chosen + landsat_date_chosen)
order = np.argsort(ind)
ind.sort()
order = order + 1
Init_Modis = 1
Init_LandSat = len(modis_date_chosen) + 1

# LandSat Image not Chosen
landsat_date_chosen_index = [int((landsat_date == x).nonzero()[0][0]) for x in landsat_date_nochosen]
K = len(landsat_date_chosen_index)
landsat_band_chosen_ind = list(range(len(landsat_band_chosen)))
observation_landsat_nochosen = torch.zeros(
    (x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(landsat_band_chosen) * scale_L ** 2, K)
for iband in range(len(landsat_band_chosen_ind)):
    for i in range(K):
        y_landsat_temp = []
        Image_landsat_temp = []
        Image_landsat = landsat[landsat_date_chosen_index[i], landsat_band_chosen_ind[iband], :, :]
        Y_landsat = Image_landsat[y_lims[0] * scale_L:(y_lims[1] + 1) * scale_L,
                    x_lims[0] * scale_L:(x_lims[1] + 1) * scale_L]
        observation_landsat_nochosen[iband * Y_landsat.numel(): (iband + 1) * Y_landsat.numel(),
        i:i + 1] = torch.reshape(
            Y_landsat, [Y_landsat.numel(), 1])

# Normalization factor estimation
# Modis Image Chosen
modis_date_chosen = list(set(landsat_date_chosen + landsat_date_nochosen).intersection(modis_date_chosen))
modis_date_chosen.sort()
modis_date_chosen_index = [int((modis_date == x).nonzero()[0][0]) for x in modis_date_chosen]
K = len(modis_date_chosen_index)
modis_band_chosen_ind = list(range(len(modis_band_chosen)))
observation_Modis_chosen = torch.zeros(
    (x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(modis_band_chosen), K)
D_matrix = torch.zeros((x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(modis_band_chosen), K)
for iband in range(len(modis_band_chosen_ind)):
    for i in range(K):
        y_Modis_temp = []
        Image_Modis_temp = []
        Image_Modis = modis[modis_date_chosen_index[i], modis_band_chosen_ind[iband], :, :]
        Y_Modis = Image_Modis[y_lims[0]:y_lims[1] + 1, x_lims[0]:x_lims[1] + 1]
        observation_Modis_chosen[iband * Y_Modis.numel(): (iband + 1) * Y_Modis.numel(), i:i + 1] = torch.reshape(
            Y_Modis, [Y_Modis.numel(), 1])

# Landsat Image Chosen
landsat_date_chosen = modis_date_chosen
landsat_date_chosen_index = [int((landsat_date == x).nonzero()[0][0]) for x in landsat_date_chosen]
K = len(landsat_date_chosen_index)
landsat_band_chosen_ind = list(range(len(landsat_band_chosen)))
observation_landsat_all = torch.zeros(
    (x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(landsat_band_chosen) * scale_L ** 2, K)
for iband in range(len(landsat_band_chosen_ind)):
    for i in range(K):
        y_landsat_temp = []
        Image_landsat_temp = []
        Image_landsat = landsat[landsat_date_chosen_index[i], landsat_band_chosen_ind[iband], :, :]
        Y_landsat = Image_landsat[y_lims[0] * scale_L:(y_lims[1] + 1) * scale_L,
                    x_lims[0] * scale_L:(x_lims[1] + 1) * scale_L]
        observation_landsat_all[iband * Y_landsat.numel(): (iband + 1) * Y_landsat.numel(), i:i + 1] = torch.reshape(
            Y_landsat, [Y_landsat.numel(), 1])

modis_est = torch.zeros(observation_Modis_chosen.size())
for i in range(observation_Modis_chosen.size(1)):
    modis_est[:, i:i + 1] = H_Modis.H_left(observation_landsat_all[:, i])
norm_factor = torch.sum(modis_est * observation_Modis_chosen) / torch.sum(modis_est ** 2)

# LandSat Image Generation for Q estimation
date_str_list = imReader.get_LandSatdate_list_from_files(dir_landsat_Qest, root_str_landsat_Qest)
date_str_list = imReader.unique(date_str_list)
landsat_imgs = []
shape = [x * scale_L for x in modis_size]
for date in date_str_list:
    landsat_imgs.append(imReader.ReadLandsat8(dir_landsat_Qest, root_str_landsat_Qest, date, shape))

landsat_size = landsat_imgs[0].shape
landsat = torch.zeros(landsat_imgs.__len__(), landsat_size[0], landsat_size[1], landsat_size[2])
landsat_date = torch.zeros(landsat_imgs.__len__(), 1, dtype=int)
for i in range(landsat_imgs.__len__()):
    for j in range(landsat_size[0]):
        bandindice = 'B' + str(j + 4)
        landsat[i, j, :, :] = torch.as_tensor(landsat_imgs[i].img_dict[bandindice])
    landsat_date[i, :] = int(landsat_imgs[i].date)

# LandSat Image Chosen
K = landsat.size(0)
landsat_band_chosen_ind = list(range(len(landsat_band_chosen)))
observation_landsat_all = torch.zeros(
    (x_lims[1] - x_lims[0] + 1) * (y_lims[1] - y_lims[0] + 1) * len(landsat_band_chosen) * scale_L ** 2, K)
for iband in range(len(landsat_band_chosen_ind)):
    for i in range(K):
        y_landsat_temp = []
        Image_landsat_temp = []
        Image_landsat = landsat[i, landsat_band_chosen_ind[iband], :, :]
        Y_landsat = Image_landsat[y_lims[0] * scale_L:(y_lims[1] + 1) * scale_L,
                    x_lims[0] * scale_L :(x_lims[1] + 1) * scale_L]
        observation_landsat_all[iband * Y_landsat.numel(): (iband + 1) * Y_landsat.numel(),
        i:i + 1] = torch.reshape(Y_landsat, [Y_landsat.numel(), 1])

mdic = {"observation_Modis": observation_Modis.double().numpy(),
        "observation_LandSat": observation_landsat.double().numpy(),
        "observation_LandSat_nochosen": observation_landsat_nochosen.double().numpy(),
        "order": order, "Init_Modis": Init_Modis, "Init_LandSat": Init_LandSat,
        "ind": ind,
        # "landsat_date_nochosen":landsat_date_nochosen,
        "ind_matrix": ind_matrix.double().numpy(),
        "norm_factor": norm_factor.double().numpy(),
        "observation_landsat_all": observation_landsat_all.double().numpy()}
savemat(Originaldatafname, mdic)
