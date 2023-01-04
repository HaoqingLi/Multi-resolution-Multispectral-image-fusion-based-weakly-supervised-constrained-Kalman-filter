import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt
import torch
import numpy as np
import scipy.io as io
from ConfigureFile import *


def ReconstructionPlot(x_kf_est, x_sm_est):
    datafile = io.loadmat(Originaldatafname)  # key is 'data_temp'
    y_LandSat = torch.as_tensor(datafile['observation_LandSat']).float()
    y_Modis = torch.as_tensor(datafile['observation_Modis']).float()
    ind = torch.as_tensor(np.double(datafile['ind'])).int()
    order = torch.as_tensor(np.double(datafile['order'])).float()
    Init_Modis = int(np.double(datafile['Init_Modis']))
    Init_LandSat = int(np.double(datafile['Init_LandSat']))
    ind_matrix = torch.as_tensor(datafile['ind_matrix'].astype(int))
    y_LandSat_nochosen = torch.as_tensor(datafile['observation_LandSat_nochosen']).float()
    landsat_date_nochosen = torch.as_tensor(datafile['landsat_date_nochosen']).int()[0]
    norm_factor = torch.as_tensor(datafile["norm_factor"]).float()
    y_LandSat_all = torch.cat((y_LandSat, y_LandSat_nochosen), dim = 1).float()
    y_LandSat = y_LandSat[:, 1:y_LandSat.size(1)]


    N = ind.size(0)*ind.size(1)
    ind_sort = torch.unique(ind).int()
    M = ind_sort.size(0)
    font_size = 14
    n_rows = 4
    n_cols = M+2-1
    n_bands = len(modis_band_chosen)
    date_ind = [ind_sort.tolist().index(x) for x in ind.tolist()[0]]

    landsatsize = [int(np.sqrt(y_LandSat.size(0)/n_bands)), int(np.sqrt(y_LandSat.size(0)/n_bands))]
    modissize = [int(np.sqrt(y_Modis.size(0)/n_bands)), int(np.sqrt(y_Modis.size(0)/n_bands))]

    Label = ["Landsat", "MODIS", "KF","Smoother"]

    for iband in range(n_bands):
        pixelmax = np.max([torch.max(y_LandSat[iband * landsatsize[0] * landsatsize[1]:(iband + 1) * landsatsize[0] * landsatsize[1], :]).numpy(),
                           torch.max(y_LandSat[iband * landsatsize[0] * landsatsize[1]:(iband + 1) * landsatsize[0] * landsatsize[1], :]).numpy(),
                           torch.max(y_Modis[iband * modissize[0] * modissize[1]: (iband + 1) * modissize[0] * modissize[1],:]).numpy()])
        pixelmax = pixelmax - 0.1
        pixelmin = 0
        landsat_count = -1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6))
        axs = axs.ravel()
        for i in range(1, N):
            if order[0, i] >= Init_LandSat:
                landsat_count += 1
                axsind = 0 * n_cols + date_ind[i]+landsat_count
                tax = axs[axsind].imshow(torch.reshape(y_LandSat[iband * landsatsize[0] * landsatsize[1]:
                                                                  (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       int(order[0, i]) - Init_LandSat],
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')
            else:
                axsind = 1 * n_cols + date_ind[i]+landsat_count
                tax = axs[axsind].imshow(torch.reshape(y_Modis[iband * modissize[0] * modissize[1]:
                                                                 (iband + 1) * modissize[0] * modissize[1],
                                                       int(order[0, i]) - Init_Modis],
                                                       [modissize[0], modissize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')
            if (landsat_date_nochosen == ind[0][i]).nonzero(as_tuple=True)[0].nelement() != 0:
                # landsat_count += 1
                axsind = 0 * n_cols + date_ind[i]+landsat_count
                index = (landsat_date_nochosen == ind[0][i]).nonzero(as_tuple=True)[0]
                tax = axs[axsind].imshow(torch.reshape(y_LandSat_nochosen[iband * landsatsize[0] * landsatsize[1]:
                                                                 (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       index],
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')


            axsind = 2 * n_cols + date_ind[i]+landsat_count
            tax = axs[axsind].imshow(torch.reshape(x_kf_est[iband * landsatsize[0] * landsatsize[1]:
                                                           (iband + 1) * landsatsize[0] * landsatsize[1],
                                                   i],
                                                   [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                     cmap='gray', interpolation='nearest', aspect='auto')

            axsind = 3 * n_cols + date_ind[i]+landsat_count
            tax = axs[axsind].imshow(torch.reshape(x_sm_est[iband * landsatsize[0] * landsatsize[1]:
                                                            (iband + 1) * landsatsize[0] * landsatsize[1],
                                                   i],
                                                   [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                     cmap='gray', interpolation='nearest', aspect='auto')

            axs[axsind].set_xlabel(r"$k = $" + str(i), fontsize=20)
        landsat_count = 0
        for i in range(1, N):
            axsind = 0 * n_cols + date_ind[i]+landsat_count
            if (landsat_date_nochosen == ind[0][i]).nonzero(as_tuple=True)[0].nelement() != 0:
                date = str(int(ind[0][i].numpy() - 20180000))
                if len(date) == 3:
                    date = str(0) + date[0] + "/" + date[1:len(date)]
                else:
                    date = date[0:2] + "/" + date[2:len(date)]
                axs[axsind].set_title(date + "M", fontsize=font_size, fontweight="bold")
            else:
                if i + 1 > N - 1:
                    ind_end = 0
                else:
                    ind_end = i+1
                if date_ind[i] - date_ind[ind_end] == 0 or date_ind[i] - date_ind[i-1] == 0:
                    date = str(int(ind[0][i].numpy() - 20180000))
                    if len(date) == 3:
                        date = str(0) + date[0] + "/" + date[1:len(date)]
                    else:
                        date = date[0:2] + "/" + date[2:len(date)]
                    if date_ind[i] - date_ind[ind_end] == 0:
                        axs[axsind].set_title(date + "M", fontsize=font_size, fontweight="bold")
                    else:
                        axs[axsind].set_title(date + "L", fontsize=font_size, fontweight="bold")
                    if date_ind[i] - date_ind[ind_end] == 0:
                        landsat_count += 1
                else:
                    date = str(int(ind[0][i].numpy() - 20180000))
                    if len(date) == 3:
                        date = str(0) + date[0] + "/" + date[1:len(date)]
                    else:
                        date = date[0:2] + "/" + date[2:len(date)]
                    axs[axsind].set_title(date + "M", fontsize=font_size, fontweight="bold")
        for irows in range(n_rows):
            axs[irows * n_cols].set_ylabel(Label[irows], rotation="vertical", fontsize=20, fontdict=dict(weight='bold'))
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(right=0.94, left=0.05, top=0.9, bottom=0.05)
        cbar_ax = fig.add_axes([0.95, 0.05, 0.01, 0.85])
        fig.colorbar(tax, cax=cbar_ax)
        figrename = 'Reconstruction_' + 'band_' + str(modis_band_chosen[iband])
        plt.savefig(figrename)
        plt.show()


    n_rows_error = n_rows-2
    n_cols_error = y_LandSat.size(1) -1 + y_LandSat_nochosen.size(1)
    Label_error = ["KF","Smoother"]
    for iband in range(n_bands):
        date_error = []
        pixelmax = np.max([torch.max(y_LandSat[iband * landsatsize[0] * landsatsize[1]:(iband + 1) * landsatsize[0] * landsatsize[1], :]).numpy(),
                           torch.max(y_LandSat[iband * landsatsize[0] * landsatsize[1]:(iband + 1) * landsatsize[0] * landsatsize[1], :]).numpy(),
                           torch.max(y_Modis[iband * modissize[0] * modissize[1]: (iband + 1) * modissize[0] * modissize[1],:]).numpy()])
        pixelmax = pixelmax - 0.1
        pixelmin = 0
        ind_NDVI = 0
        fig, axs = plt.subplots(n_rows_error, n_cols_error, figsize=(16, 5))
        axs = axs.ravel()
        for i in range(N-1):
            if order[0, i + 1] > Init_LandSat:
                axsind = 0 * n_cols_error + ind_NDVI
                date = str(int(ind[0][i].numpy() - 20180000))
                if len(date) == 3:
                    date = str(0) + date[0] + "/" + date[1:len(date)]
                else:
                    date = date[0:2] + "/" + date[2:len(date)]
                date_error.append(date)
                axs[axsind].set_title(date, fontsize=font_size, fontweight="bold")
                tax = axs[axsind].imshow(torch.reshape(torch.abs(x_kf_est[iband * landsatsize[0] * landsatsize[1]:
                                                                (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       i ] - y_LandSat[iband * landsatsize[0] * landsatsize[1]:
                                                                  (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       int(order[0, i+1]) - Init_LandSat]),
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')
                axsind = 1 * n_cols_error + ind_NDVI
                tax = axs[axsind].imshow(torch.reshape(torch.abs(x_sm_est[iband * landsatsize[0] * landsatsize[1]:
                                                                (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       i]- y_LandSat[iband * landsatsize[0] * landsatsize[1]:
                                                                  (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       int(order[0, i+1]) - Init_LandSat]),
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')

                axs[axsind].set_xlabel(r"$k = $" + str(i), fontsize=20)
                ind_NDVI += 1
            if (landsat_date_nochosen == ind[0][i]).nonzero(as_tuple=True)[0].nelement() != 0:
                index = (landsat_date_nochosen == ind[0][i]).nonzero(as_tuple=True)[0]
                axsind = 0 * n_cols_error + ind_NDVI
                date = str(int(ind[0][i].numpy() - 20180000))
                if len(date) == 3:
                    date = str(0) + date[0] + "/" + date[1:len(date)]
                else:
                    date = date[0:2] + "/" + date[2:len(date)]
                date_error.append(date)
                axs[axsind].set_title(date, fontsize=font_size, fontweight="bold")
                tax = axs[axsind].imshow(torch.reshape(torch.abs(x_kf_est[iband * landsatsize[0] * landsatsize[1]:
                                                                          (iband + 1) * landsatsize[0] * landsatsize[1],
                                                                 i:i+1] - y_LandSat_nochosen[iband * landsatsize[0] * landsatsize[1]:
                                                                 (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       index]),
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')
                axsind = 1 * n_cols_error + ind_NDVI
                tax = axs[axsind].imshow(torch.reshape(torch.abs(x_sm_est[iband * landsatsize[0] * landsatsize[1]:
                                                                          (iband + 1) * landsatsize[0] * landsatsize[1],
                                                                 i:i+1] - y_LandSat_nochosen[iband * landsatsize[0] * landsatsize[1]:
                                                                 (iband + 1) * landsatsize[0] * landsatsize[1],
                                                       index]),
                                                       [landsatsize[0], landsatsize[1]]), vmin=pixelmin, vmax=pixelmax,
                                         cmap='gray', interpolation='nearest', aspect='auto')
                axs[axsind].set_xlabel(r"$k = $" + str(i), fontsize=20)
                ind_NDVI += 1
        for irows in range(n_rows_error):
            axs[irows * n_cols_error].set_ylabel(Label_error[irows], rotation="vertical", fontsize=20, fontdict=dict(weight='bold'))
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(right=0.94, left=0.05, top=0.9, bottom=0.05)
        cbar_ax = fig.add_axes([0.95, 0.05, 0.01, 0.85])
        fig.colorbar(tax, cax=cbar_ax)
        figrename = 'ReconstructionError_' + 'band_' + str(modis_band_chosen[iband])
        plt.savefig(figrename)
        plt.show()

