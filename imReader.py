import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from abc import ABC, abstractmethod
import rasterio
from rasterio.plot import show as rshow
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


class ReadImg(ABC):
    """
    Abstract class eadImg. All compatible image readers must inherent from this class and implement the abstract
    methods.
    Contains abstract methods:
      - read_img(self, filename)
      - save_img(self, filename)
      - read_qc(self, filename)
      - plot_img(self, o_raw_img=False)
    Attributes:
        - self.data_type_scale_dict = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64,
                                     'int8': 7, 'int16': 15, 'int32': 31, 'int64': 63}
    """

    def __init__(self):
        """
        Initialize an abstract ReadImg model
        """
        super().__init__()

        self.data_type_scale_dict = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64,
                                     'int8': 7, 'int16': 15, 'int32': 31, 'int64': 63}

    @abstractmethod
    def read_img(self, filename):
        """
        Abstract method to read image.
        :param filename: String with file name with path.
        :param format: string with file formats
        :return: None
        """

    @abstractmethod
    def save_img(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """

    @abstractmethod
    def read_qc(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """

    @abstractmethod
    def plot_img(self, o_raw_img=False):
        """
        Plot image
        :param o_raw_img: Boolean variable. If True plot raw image with original values. If False plot image scaled
        ([0,1]) image.
        :return: None
        """

    def __str__(self):
        """
        :return:
        """
        return str(self.__dict__)


class ReadMODIS(ReadImg):
    """
    This class creates a multiple band object intended to read and store multi-band 500m MODIS images.
    It stores the images in a dictionary containing one numpy 2D array for each band. Every band is assumed to have
    the same shape (num of rows and columns) and to be represented with the same numerical format.
    QA codes are also processed following the coding convention in ...
    MODIS QA for 500m channels code starts from the least significative bit (LSB)
        bits 0-1: MODLAND QA bit
             00: corrected product produced at ideal quality -- all bands
             01: corrected product produced at less than ideal quality -- some or all bands
             10: corrected product not produced due to cloud effects -- all bands
             11: corrected product not produced for other reasons -- some or all bands, may be fill value (11)
             [Note that a value of (11) overrides a value of (01)]
        bits 2-5 band 1 data quality, four bit range
            0000 highest quality
            0111 noisy detector
            1000 dead detector, data interpolated in L1B
            1001 solar zenith >= 86 degrees
            1010 solar zenith >= 85 and < 86 degrees
            1011 missing input
            1100 internal constant used in place of climatological data for at
            least one atmospheric constant
            1101 correction out of bounds, pixel constrained to extreme allowable value
            1110 L1B data faulty
            1111 not processed due to deep ocean or clouds
        bits 6-9 band 2 data quality four bit range same as band above
        bits 10-13 band 3 data quality four bit range same as band above
        bits 14-17 band 4 data quality four bit range same as band above
        bits 18-21 band 5 data quality four bit range same as band above
        bits 22-25 band 6 data quality four bit range same as band above
        bits 26-29 band 7 data quality four bit range same as band above
        bit 30 atmospheric correction performed
            1 yes
            0 no
        bit 31 adjacency correction performed:
            1 yes
            0 no
        When processing QA codes we store boolean variable masks. If true it indicates that a particular pixel in a
        particular band is usable. If False, then the pixel is somewhat corrupted. For mor info on these definition read
        the class static methods:
         - eval_qa(qc_image)
         - eval_band_qa(qc_image, band_index)
        The main attributes of this class are:
        self.band_indexes = [1, 2, 3, 4, 5, 6, 7]
        self.band_strs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        self.band_index_dict = {'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'B6': 6, 'B7': 7}
        self.qc_str = 'QC'
        self.date = ''
        self.img_dict: dictionary containing the image data for bands existing in the directory.
        self.shape: tuple with (number of bands, number of rows, number of columns).
        self.qa: boolean mask resulting from processing the MODLAND QA bit (providing info for all bands)
        self.qa_per_band: dictionary boolean mask resulting from processing QA bits for each individual band existing
                          in the directory.
        self.scale_factor = the scaling adopted when converting the raw images int16 images to float64.
    """

    def __init__(self, dir, root_str, date_str):
        """
        This read MODIS images with the following naming convention: CommonRoot_BandIndicator_Date.tif, e.g.,
             MODIS_MYD09GA_B5_2020_09_02.tif
        This class also reads QA codes from QC files with the same name convention as image files exchanging the
        'BandIndicator' for 'QC'. That is, CommonRoot_QC_Date.tif
        Reading QA codes
        :param dir: path to directory, e.g., ls '/home/vangog/tmp/HD-Img/MYD500m-20210402T120042Z-001/MYD500m/'
        :param root_str: string with root of filenames, e.g., 'MODIS_MYD09GA_'
        :param date_str: date string, e.g., '2020_10_24'
        """
        super().__init__()

        # class attributes:
        self.band_indexes = [1, 2, 3, 4, 5, 6, 7]
        self.band_strs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        self.band_index_dict = {'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'B6': 6, 'B7': 7}

        self.qc_str = 'QC'
        self.img_dict = None
        # self.shape = None
        self.qa = None
        self.qa_per_band = {}
        self.scale_factor = None
        self.date = date_str

        [self.img_dict, qc_img, self.scaling_factor, self.shape, self.dtype, self.nbits] = \
            self.read_images_from_dir(dir, root_str, date_str)

        #self.qa = self.eval_qa(qc_img)

        # for band in self.band_strs:
        #     self.qa_per_band[band] = self.eval_band_qa(qc_img, self.band_index_dict[band])

    def read_images_from_dir(self, dir, root_str, date_str):
        """
        :param dir:
        :param root_str:
        :param date_str:
        :return:
        """

        file_str = os.path.join(dir, root_str + '*' + date_str + '*.tif')
        file_list = glob.glob(file_str)
        file_list.sort()
        img_dict = {}
        qc_img = None
        scaling = None
        shape = None
        dtype = None
        nbits = None
        for file in file_list:
            # read file
            # is a QC file?
            if 'QC' in file:
                qc_img = self.read_qc(file)
            else:
                # get band string
                tokens = file.split('_')
                band_bool = [x in tokens for x in self.band_strs]
                band_idx = [idx for idx, x in enumerate(band_bool) if x is True]
                assert len(band_idx) == 1
                band_srt = self.band_strs[band_idx[0]]
                # read img file
                [scaled_img_data, scaling, shape, dtype, nbits] = self.read_img(file)
                # storing on img dictionary
                img_dict[band_srt] = scaled_img_data
        nb = len(img_dict)
        shape = (nb, shape[1], shape[2])
        return [img_dict, qc_img, scaling, shape, dtype, nbits]

    @staticmethod
    def eval_qa(qc_image):
        """
        Decode QA bits from bit string
        :param sbit: string with QC 32 bit chain.
        :return: boolean variable. If True bit string is
                    00: corrected product produced at ideal quality -- all bands
                or
                    01: corrected product produced at less than ideal quality -- some or all bands
                if False bit string is
                    10: corrected product not produced due to cloud effects -- all bands
                or
                    11: corrected product not produced for other reasons -- some or all bands, may be fill value (11)
        """
        qa_mask_mat = np.zeros(qc_image.shape, np.bool)

        for i in range(qc_image.shape[0]):
            for j in range(qc_image.shape[1]):
                if (qc_image[i, j] & 0x03) <= 1:
                    qa_mask_mat[i, j] = True
        return qa_mask_mat

    @staticmethod
    def eval_band_qa(qc_image, band_index):
        """
        :param self:
        :param qc_image:
        :param band_index:
        :return:
        """
        qa_mask_mat = np.zeros(qc_image.shape, np.bool)
        for i in range(qc_image.shape[0]):
            for j in range(qc_image.shape[1]):
                pixel_code = (qc_image[i, j] >> (2 + 4 * (band_index - 1))) & 0x0F
                if pixel_code == 0:
                    qa_mask_mat[i, j] = True
        return qa_mask_mat

    def read_img(self, filename):
            """
            Abstract method to read image.
            :param filename: String with file name with path.
            :return: [scaled_img_data, scaling, shape, dtype, nbits]
            """

            rasterio_ = rasterio.open(filename)
            dtype = rasterio_.dtypes[0]
            assert dtype in self.data_type_scale_dict
            nbits = self.data_type_scale_dict[dtype]
            scaling = 2 ** (nbits-2)

            raw_img_data = rasterio_.read()
            scaled_img_data = (1/scaling)*raw_img_data
            shape = scaled_img_data.shape
            if len(shape) > 2:
                if shape[0] == 1:
                    scaled_img_data = scaled_img_data.reshape(shape[1], shape[2])

            return [scaled_img_data, scaling, shape, dtype, nbits]

    def save_img(self, filename):
        """
        Abstract method to save image.
        :param filename: String with TIF file name with path.
        :return: None
        """

    def read_qc(self, filename):
        """
        :param filename: String with file name with path.
        :return: None
        """
        qc_ = rasterio.open(filename)
        # self.n_qc_bits = self.data_type_scale_dict[qc_.dtypes[0]]
        qc_ = qc_.read()
        n_rows = qc_.shape[1]
        n_cols = qc_.shape[2]
        qc = qc_.reshape(n_rows, n_cols)
        return qc
        # qa = self.eval_qa(self.qc)
        # for band_index in self.band_indexes:
        # for band in self.band_index_dict.keys():
        #     self.qa_per_band[band] = self.eval_band_qc(qc, self.band_index_dict[band])

        # binary_qc = [np.binary_repr(self.qc[i][j]) for i in self.qc]

    def plot_img(self, band_key=None, n_cols=2):
        """
        Plot image
        :param o_raw_img: Boolean variable. If True plot raw image with original values. If False plot image scaled
        ([0,1]) image.
        :return: None
        """

        font_size = 12
        n_rows = round(self.shape[0]/2 + 0.5)
        # plt.figure()

        # fig, axs = plt.subplots(n_rows, 2*n_cols, sharex=True, sharey=True)
        fig = plt.figure(figsize=(14, 8))
        axs = fig.subplots(n_rows, 2*n_cols, sharex=True, sharey=True)
        axs = axs.ravel()

        if band_key is None:
            #plot all bands
            count = 1
            for key in self.img_dict.keys():
                # ax = plt.subplot(n_rows, n_cols, count)
                # tax = rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                # rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                tax = axs[count - 1].imshow(self.img_dict[key], cmap='gray')
                fig.colorbar(tax, ax=axs[count - 1])
                axs[count - 1].set_title(key, fontsize=font_size)
                # tax_qa = rshow(self.qa_per_band[key], ax=axs[n_cols*n_rows + count - 1], cmap='gray', vmin=0, vmax=1)
                tax_qa = axs[n_cols * n_rows + count - 1].imshow(self.qa_per_band[key], cmap='gray', vmin=0, vmax=1)
                axs[n_cols*n_rows + count - 1].set_title('QA ' + key, fontsize=font_size)
                fig.colorbar(tax_qa, ax=axs[n_cols*n_rows + count - 1])
                count += 1
            ta = axs[count - 1].imshow(self.qa, cmap='gray', vmin=0, vmax=1)
            axs[count - 1].set_title('QA', fontsize=font_size)
            fig.colorbar(ta, ax=axs[count - 1])
        else:
            rshow(self.img_dict[band_key])
        fig.tight_layout()
        axs[-1].set_visible(False)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.show()
        # if o_raw_img:
        #     rshow(self.raw_img_data)
        # else:
        #     rshow(self.scaled_img_data)


class ReadLandsat8(ReadImg):
    def __init__(self, dir, root_str, date_str, shape):
        """
        This read MODIS images with the following naming convention: CommonRoot_BandIndicator_Date.tif, e.g.,
             MODIS_MYD09GA_B5_2020_09_02.tif
        This class also reads QA codes from QC files with the same name convention as image files exchanging the
        'BandIndicator' for 'QC'. That is, CommonRoot_QC_Date.tif
        Reading QA codes
        :param dir: path to directory, e.g., ls '/home/vangog/tmp/HD-Img/MYD500m-20210402T120042Z-001/MYD500m/'
        :param root_str: string with root of filenames, e.g., 'MODIS_MYD09GA_'
        :param date_str: date string, e.g., '2020_10_24'
        """
        super().__init__()

        # class attributes:
        self.band_indexes = [1, 2, 3, 4, 5, 6, 7]
        self.band_strs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        self.band_index_dict = {'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'B6': 6, 'B7': 7}
        self.qc_str = 'QC'
        self.img_dict = None
        # self.shape = None
        self.qa = None
        self.qa_per_band = {}
        self.scale_factor = None
        self.date = date_str
        self.shape = shape

        [self.img_dict, qc_img, self.scaling_factor, self.shape, self.dtype, self.nbits] = \
            self.read_images_from_dir(dir, root_str, date_str)

    def read_images_from_dir(self, dir, root_str, date_str):
        """
        :param dir:
        :param root_str:
        :param date_str:
        :return:
        """

        file_str = os.path.join(dir, root_str + '*' + date_str + '*.tif')
        file_list = glob.glob(file_str)
        file_list.sort()
        img_dict = {}
        qc_img = None
        scaling = None
        shape = None
        dtype = None
        nbits = None
        for file in file_list:
            # read file
            # is a QC file?
            if 'QC' in file:
                qc_img = self.read_qc(file)
            else:
                # get band string
                tokens = file.split('_')
                band_bool = [x in tokens for x in self.band_strs]
                band_idx = [idx for idx, x in enumerate(band_bool) if x is True]
                assert len(band_idx) == 1
                band_srt = self.band_strs[band_idx[0]]
                # read img file
                [scaled_img_data, scaling, shape, dtype, nbits] = self.read_img(file)
                shape = self.shape
                scaled_img_data = np.array(Image.fromarray(scaled_img_data).resize((shape[2], shape[1]), resample = 2))
                # storing on img dictionary
                img_dict[band_srt] = scaled_img_data
        nb = len(img_dict)
        shape = (nb, shape[1], shape[2])
        return [img_dict, qc_img, scaling, shape, dtype, nbits]


    def read_img(self, filename):
        """
        Abstract method to read image.
        :param filename: String with file name with path.
        :param format: string with file formats
        :return: None
        """
        rasterio_ = rasterio.open(filename)
        dtype = rasterio_.dtypes[0]
        assert dtype in self.data_type_scale_dict
        nbits = self.data_type_scale_dict[dtype]
        scaling = 2 ** (nbits - 2)

        raw_img_data = rasterio_.read()
        scaled_img_data = (1 / scaling) * raw_img_data
        shape = scaled_img_data.shape
        if len(shape) > 2:
            if shape[0] == 1:
                scaled_img_data = scaled_img_data.reshape(shape[1], shape[2])

        return [scaled_img_data, scaling, shape, dtype, nbits]


    def save_img(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """
        pass

    def read_qc(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """
        pass

    def plot_img(self, band_key=None, n_cols=2):
        """
        Plot image
        :param o_raw_img: Boolean variable. If True plot raw image with original values. If False plot image scaled
        ([0,1]) image.
        :return: None
        """
        font_size = 12
        n_rows = round(self.shape[0] / 2 + 0.5)
        # plt.figure()

        # fig, axs = plt.subplots(n_rows, 2*n_cols, sharex=True, sharey=True)
        fig = plt.figure(figsize=(14, 8))
        axs = fig.subplots(n_rows, 2 * n_cols, sharex=True, sharey=True)
        axs = axs.ravel()

        if band_key is None:
            # plot all bands
            count = 1
            for key in self.img_dict.keys():
                # ax = plt.subplot(n_rows, n_cols, count)
                # tax = rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                # rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                tax = axs[count - 1].imshow(self.img_dict[key], cmap='gray')
                fig.colorbar(tax, ax=axs[count - 1])
                axs[count - 1].set_title(key, fontsize=font_size)
                # tax_qa = rshow(self.qa_per_band[key], ax=axs[n_cols*n_rows + count - 1], cmap='gray', vmin=0, vmax=1)
                tax_qa = axs[n_cols * n_rows + count - 1].imshow(self.qa_per_band[key], cmap='gray', vmin=0, vmax=1)
                axs[n_cols * n_rows + count - 1].set_title('QA ' + key, fontsize=font_size)
                fig.colorbar(tax_qa, ax=axs[n_cols * n_rows + count - 1])
                count += 1
            ta = axs[count - 1].imshow(self.qa, cmap='gray', vmin=0, vmax=1)
            axs[count - 1].set_title('QA', fontsize=font_size)
            fig.colorbar(ta, ax=axs[count - 1])
        else:
            rshow(self.img_dict[band_key])
        fig.tight_layout()
        axs[-1].set_visible(False)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.show()


class ReadSentinel2(ReadImg):
    def __init__(self, dir, root_str, date_str, shape):
        """
        This read MODIS images with the following naming convention: CommonRoot_BandIndicator_Date.tif, e.g.,
             MODIS_MYD09GA_B5_2020_09_02.tif
        This class also reads QA codes from QC files with the same name convention as image files exchanging the
        'BandIndicator' for 'QC'. That is, CommonRoot_QC_Date.tif
        Reading QA codes
        :param dir: path to directory, e.g., ls '/home/vangog/tmp/HD-Img/MYD500m-20210402T120042Z-001/MYD500m/'
        :param root_str: string with root of filenames, e.g., 'MODIS_MYD09GA_'
        :param date_str: date string, e.g., '2020_10_24'
        """
        super().__init__()

        # class attributes:
        self.band_indexes = [1, 2, 3, 4, 5, 6, 7]
        self.band_strs = ['B1', 'B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        self.band_index_dict = {'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B8A': 5, 'B11': 6, 'B12': 7}
        self.qc_str = 'QC'
        self.img_dict = None
        # self.shape = None
        self.qa = None
        self.qa_per_band = {}
        self.scale_factor = None
        self.date = date_str
        self.shape = shape

        [self.img_dict, qc_img, self.scaling_factor, self.shape, self.dtype, self.nbits] = \
            self.read_images_from_dir(dir, root_str, date_str)



    def read_images_from_dir(self, dir, root_str, date_str):
        """
        :param dir:
        :param root_str:
        :param date_str:
        :return:
        """

        file_str = os.path.join(dir, root_str + '*' + date_str + '*_T10SFJ.tif')
        file_list = glob.glob(file_str)
        # if any(['_T10SFJ' in x for x in file_list]):
        #     file_str = os.path.join(dir, root_str + '*' + date_str + '*_T10SFJ.tif')
        #     file_list = glob.glob(file_str)
        file_list.sort()
        img_dict = {}
        qc_img = None
        scaling = None
        shape = None
        dtype = None
        nbits = None
        for file in file_list:
            # read file
            # is a QC file?
            if 'QC' in file:
                qc_img = self.read_qc(file)
            else:
                # get band string
                tokens = file.split('_')
                band_bool = [x in tokens for x in self.band_strs]
                band_idx = [idx for idx, x in enumerate(band_bool) if x is True]
                assert len(band_idx) == 1
                band_srt = self.band_strs[band_idx[0]]
                # read img file
                [scaled_img_data, scaling, shape, dtype, nbits] = self.read_img(file)
                shape = self.shape
                scaled_img_data = np.array(Image.fromarray(scaled_img_data).resize((shape[2], shape[1]), resample=2))
                # storing on img dictionary
                img_dict[band_srt] = scaled_img_data
        nb = len(img_dict)
        shape = (nb, shape[1], shape[2])
        return [img_dict, qc_img, scaling, shape, dtype, nbits]


    def read_img(self, filename):
        """
        Abstract method to read image.
        :param filename: String with file name with path.
        :param format: string with file formats
        :return: None
        """
        rasterio_ = rasterio.open(filename)
        dtype = rasterio_.dtypes[0]
        assert dtype in self.data_type_scale_dict
        nbits = self.data_type_scale_dict[dtype]
        scaling = 2 ** (nbits - 2)

        raw_img_data = rasterio_.read()
        scaled_img_data = (1 / scaling) * raw_img_data
        shape = scaled_img_data.shape
        if len(shape) > 2:
            if shape[0] == 1:
                scaled_img_data = scaled_img_data.reshape(shape[1], shape[2])

        return [scaled_img_data, scaling, shape, dtype, nbits]


    def save_img(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """
        pass

    def read_qc(self, filename):
        """
        Abstract method to save image.
        :param filename: String with file name with path.
        :return: None
        """
        pass

    def plot_img(self, band_key=None, n_cols=2):
        """
        Plot image
        :param o_raw_img: Boolean variable. If True plot raw image with original values. If False plot image scaled
        ([0,1]) image.
        :return: None
        """
        font_size = 12
        n_rows = round(self.shape[0] / 2 + 0.5)
        # plt.figure()

        # fig, axs = plt.subplots(n_rows, 2*n_cols, sharex=True, sharey=True)
        fig = plt.figure(figsize=(14, 8))
        axs = fig.subplots(n_rows, 2 * n_cols, sharex=True, sharey=True)
        axs = axs.ravel()

        if band_key is None:
            # plot all bands
            count = 1
            for key in self.img_dict.keys():
                # ax = plt.subplot(n_rows, n_cols, count)
                # tax = rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                # rshow(self.img_dict[key], ax=axs[count - 1], cmap='gray')
                tax = axs[count - 1].imshow(self.img_dict[key], cmap='gray')
                fig.colorbar(tax, ax=axs[count - 1])
                axs[count - 1].set_title(key, fontsize=font_size)
                # tax_qa = rshow(self.qa_per_band[key], ax=axs[n_cols*n_rows + count - 1], cmap='gray', vmin=0, vmax=1)
                tax_qa = axs[n_cols * n_rows + count - 1].imshow(self.qa_per_band[key], cmap='gray', vmin=0, vmax=1)
                axs[n_cols * n_rows + count - 1].set_title('QA ' + key, fontsize=font_size)
                fig.colorbar(tax_qa, ax=axs[n_cols * n_rows + count - 1])
                count += 1
            ta = axs[count - 1].imshow(self.qa, cmap='gray', vmin=0, vmax=1)
            axs[count - 1].set_title('QA', fontsize=font_size)
            fig.colorbar(ta, ax=axs[count - 1])
        else:
            rshow(self.img_dict[band_key])
        fig.tight_layout()
        axs[-1].set_visible(False)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.show()




def get_date_list_from_qc_files(dir):
    qc_str = 'QC'
    file_str = os.path.join(dir, root_str + '*' + qc_str + '*.tif')
    file_list = glob.glob(file_str)
    file_list.sort()
    date_str_list = []
    for file in file_list:
        date_token = file.split(qc_str)[-1]
        date_token = date_token.split('.')[0]
        date_token = str(date_token)[1:]
        date_str_list.append(date_token)
    return date_str_list


def get_Moids_list_from_files(dir, root_str):
    file_str = os.path.join(dir, root_str + '*.tif')
    file_list = glob.glob(file_str)
    file_list.sort()
    date_str_list = []
    for file in file_list:
        date_token = file.split('.tif')[0]
        date_token = date_token.split('_')
        date_token = str(date_token[-3]) + '_' + str(date_token[-2]) + '_' + str(date_token[-1])
        date_str_list.append(date_token)
    return date_str_list

def get_LandSatdate_list_from_files(dir, root_str):
    file_str = os.path.join(dir, root_str + '*.tif')
    file_list = glob.glob(file_str)
    file_list.sort()
    date_str_list = []
    for file in file_list:
        date_token = file.split('_')[-1]
        date_token = date_token.split('.')[0]
        date_token = str(date_token)[:]
        date_str_list.append(date_token)
    return date_str_list


def get_Sentineldate_list_from_files(dir, root_str):
    file_str = os.path.join(dir, root_str + '*_T10SFJ.tif')
    file_list = glob.glob(file_str)
    file_list.sort()
    date_str_list = []
    for file in file_list:
        date_token = file.split('_')[-2]
        date_token = date_token.split('T')[0]
        date_token = str(date_token)[:]
        date_str_list.append(date_token)
    return date_str_list


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list


if __name__ == '__main__':
    print('Hello World')