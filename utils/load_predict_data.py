import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from train import parse_opt
from utils.tools import load_df, seg_data

opt = parse_opt()
np.random.seed(1234)


# length = {'Bearing1_3': 1802,
#           'Bearing1_4': 1139,
#           'Bearing1_5': 2302,
#           'Bearing1_6': 2302,
#           'Bearing1_7': 1502,
#           'Bearing2_3': 1202,
#           'Bearing2_4': 612,
#           'Bearing2_5': 2002,
#           'Bearing2_6': 572,
#           'Bearing2_7': 172,
#           'Bearing3_3': 352}

if opt.type == 'PHM':
    length = {'Bearing1_3': 1229,
            'Bearing1_4': 1105,
            'Bearing1_5': 2141,
            'Bearing1_6': 2156,
            'Bearing1_7': 745,
            'Bearing2_3': 449,
            'Bearing2_4': 473,
            'Bearing2_5': 1693,
            'Bearing2_6': 443,
            'Bearing2_7': 114,
            'Bearing3_3': 270}
    main_dir_colab = opt.main_dir_colab
else:
    length = {'Bearing1_3': 98,
              'Bearing1_4': 122,
              'Bearing1_5': 13,
              'Bearing2_3': 223,
              'Bearing2_4': 13,
              'Bearing2_5': 219,
              'Bearing3_3': 31,
              'Bearing3_4': 97,
              'Bearing4_5': 105}
    main_dir_colab = '/content/drive/MyDrive/Khoa/XJTU_data/XJTU-SY_Bearing_Datasets/'


test_data_2D   = seg_data(load_df(main_dir_colab + f'test_data_2D_{opt.condition}.pkz'), length)
test_data_1D   = seg_data(load_df(main_dir_colab + f'test_data_1D_{opt.condition}.pkz'), length)
test_data_extract   = seg_data(load_df(main_dir_colab + f'test_data_extract_{opt.condition}.pkz'), length)
test_data_c   = seg_data(load_df(main_dir_colab + f'test_c_{opt.condition}.pkz'), length)

test_label_1D  = seg_data(load_df(main_dir_colab + f'test_label_1D_{opt.condition}.pkz'), length)

