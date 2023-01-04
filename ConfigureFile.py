

#Parameter Setting

Originaldatafname = './KFSMDataset.mat'  #Original data set
x_lims = [58, 66]                        # Index for chosen MODIS image area in X axis
y_lims = [52, 60]                        # Index for chosen MODIS image area in Y axis

nl = 9                                   # image length
ns = 9                                   # image width
nb = 2                                   # bands
scale_L = 9                              # scale between Landsat and MODIS images
scale_multi_factor = 1
epsilon = 1e-8
ModelOption = 'Diagonal'
                                        #Diagonal
                                        #BlockDiagonal
                                        #Full
alpha = 0.1                             # Weight of shrunk covariance calculation for Q matrix estimation


# Modis Image Generation
dir_modis = r'C:\D_Drive\ImageFusion\SplitKF_SampleCode\MODIS_250/'
root_str_modis = 'MODIS_MYD09GQ_'
# LandSat Image Generation
dir_landsat = r'C:\D_Drive\ImageFusion\SplitKF_SampleCode/HD-IMG-Database-Landsat-8/'
root_str_landsat = 'LandSat8_*_044032*'
# LandSat Image Generation for Q estimation
dir_landsat_Qest = r'C:\D_Drive\ImageFusion\SplitKF_SampleCode/HD-IMG-Database-Landsat-8-Qest/'
root_str_landsat_Qest = 'LandSat8_*_044032*'

# Modis Image Chosen as Observations
modis_band_chosen = [1, 2]
modis_date_chosen = [20180703, 20180709, 20180714, 20180719, 20180726, 20180801, 20180803, 20180808,
                     20180813, 20180820, 20180824, 20180829, 20180905, 20180911, 20180916, 20180921]

# Landsat Image Chosen as Observations and Ground-truth
landsat_band_chosen = [4, 5]
# The first one is for state initialization and the rest are observations
landsat_date_chosen = [20180601, 20180703, 20180921]
# Ground-truth
landsat_date_nochosen = [20180719, 20180820, 20180905]

## Date interval for states and histotical data to estimate Q matrix
Historicaldates = [16, 16, 32, 64]
dates_kf = [32, 0, 6, 5, 5, 7, 6, 2, 5, 5, 7, 4, 5, 7, 6, 5, 5, 0]