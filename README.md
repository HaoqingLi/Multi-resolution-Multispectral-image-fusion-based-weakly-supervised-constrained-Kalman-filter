# Multi-resolution Multispectral image fusion based on a weakly supervised constrained Kalman filter
It's a software package for a constrained-distributed-Kalman-filter-based Multi-resolution Multispectral image fusion method. For more details about the method, please refer to the reference [1] below, also available in arXiv at https://arxiv.org/pdf/2301.02598.

If you use this software please cite the following in any resulting publication:

    [1] Online Fusion of Multi-resolution Multispectral Images with Weakly Supervised Temporal Dynamics
        Haoqing Li, Bhavya Duvvuri, Ricardo Borsoi, Tales Imbiriba, Edward Beighley, Deniz Erdoğmuş, Pau Closas
        ISPRS Journal of Photogrammetry and Remote Sensing, 2023.


# Instructions
This project is a sample code aiming to fuse Landsat image using both Landsat and MODIS image at Oroville dam site. Specifically, Landsat images at 2018/07/03 and 2018/09/21, as well as 16 MODIS images at dates from 2018/07/03 to 2018/09/21, are used to estimate the Landsat images from 2018/07/03 to 2018/09/21. Note that two bands of each kind of image are considered, band 1 and band 2 are included for MODIS image while band 4 and band 5 are included for Landsat image.

The dataset can be found in https://zenodo.org/record/7504719#.Y7Xz53bMKF4, where 3 folder are listed: one for MODIS image observations, which is “MODIS_250” in the dataset, one for Landsat image observations, which is “HD-IMG-Database-Landsat-8” in the dataset, and the final one is historical data for Q matrix estimation, which is “HD-IMG-Database-Landsat-8-Qest” in the dataset.

Input parameters are listed in ConfigureFile.py file, which would be introduced in detail as below:
 1) Originaldatafname shows the path to read images and save it as a mat file. In this mat file, “observation_Modis” shows the pixel value of MODIS image observations; “observation_LandSat” shows the pixel value of Landsat images observations while the first vector is for state initialization in KF; “observation_LandSat_nochosen” shows the Landsat images as ground-truth; “observation_landsat_all” shows the historical Landsat images for Q matrix estimation. When ordering all observations in the form as [MODIS, Landsat], where MODIS and Landsat contain corresponding observations in time series, the order would be saved as “order” and “Init_Modis” shows the first order of MOIDS observation, “Init_LandSat” shows the first order of Landsat observation. Besides, “ind” contains all dates for both MODIS and Landsat observations, “ind_matrix” contains the index for Landsat images transforming into vector, and “norm_factor” is a factor to keep the match between MOIDS image and the one decimated using Landsat image.
2) x_lims and y_lims show the X and Y index for the chosen area in MODIS image.

3) nl and ns are the length and width of MODIS image and nb is the number of bands of MOIDS/Landsat images.

4) scale_L is the scaling factor to to map MODIS and Landsat images.

5) scale_multi_factor is the scaling factor to decide size of convolution window to map MODIS and Landsat images.

6) epsilon is a small positive number used to make sure all image pixel is positive and no covariance of states in KF and smoother are positive definite.

7) ModelOption has three different options to choose: 1) “Diagonal” means the pixels in Landsat images are totallty independent; 2) “BlockDiagonal” means the pixels corresponding to the same area in Landsat images of different bands are correlated; 3) “Full” means the pixels corresponding to the same area in MODIS images of different bands are correlated.

8) alpha is a weight factor of diagonal matrix, used in shrunk covariance calculation for Q matrix estimation.

9) dir_modis shows the path of MODIS images and root_str_modis is the root string of names of MODIS images.

10) dir_landsat shows the path of Landsat images and root_str_landsat is the root string of names of Landsat images.

11) dir_landsat_Qest shows the path of historical Landsat images and root_str_landsat_Qest is the root string of names of historical Landsat images.

12) modis_band_chosen shows band of MODIS images. 

13) modis_date_chosen shows the dates of MOIDS images that used as observations in KF and smoother.

14) landsat_band_chosen shows band of Landsat images. 

15) landsat_date_chosen shows the dates of Landsat images, where the first one is for state initialization in KF, and the rest are for observations in KF and smoother.
16) landsat_date_nochosen shows dates of Landsat images that are not chosen as observations in KF and smoother, but as ground-truth for performance measurement.
17) Historicaldates is to show the date intervals in days between Landsat images in historical data set.
18) Dates_KF is to show the date intervals in days between states in KF.

The following shows how to run the code and where to find the results:
1)	Run DatasetGenerator.py 
2)	Run RealData4KFSM_Qt.py
The reconstruction images as well as the reconstruction error images are saved in the code folder as png file. The results of KF and smoother are saved as mat file in code folder with name in a form as “landsat_2band_xxx_alpha_xxx_Qt.mat”.
Note that in the reconstruction figure, we skip the results of MOIDS observations at 2018/07/03, that’s because we need to use the results to compare with ESTARFM and PSRFM method in the paper, where no estimations would acquired at 2018/07/03. 



