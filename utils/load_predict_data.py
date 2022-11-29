import numpy as np
from train import parse_opt
from utils.tools import load_df, seg_data

opt = parse_opt()
np.random.seed(1234)


if opt.type == 'PHM':
    length = {'Bearing1_1': 1314,
              'Bearing1_2': 826,
              'Bearing1_3': 1726,
              'Bearing1_4': 1082,
              'Bearing1_5': 2412,
              'Bearing1_6': 1631,
              'Bearing1_7': 2210}
    main_dir_colab = opt.main_dir_colab
else:
    length = {'Bearing1_1': 76,
              'Bearing1_2': 44,
              'Bearing1_3': 60,
              'Bearing1_5': 39,
              'Bearing2_1': 455,
              'Bearing2_2': 48,
              'Bearing2_3': 327,
              'Bearing2_4': 32,
              'Bearing2_5': 141,
              'Bearing3_1': 2344,
              'Bearing3_3': 340,
              'Bearing3_4': 1418,
              'Bearing3_5': 9}
    main_dir_colab = '/content/drive/MyDrive/Khoa/XJTU_data/XJTU-SY_Bearing_Datasets/'


test_data_2D   = seg_data(load_df(main_dir_colab + f'test_data_2D_{opt.condition}.pkz'), length)
test_data_1D   = seg_data(load_df(main_dir_colab + f'test_data_1D_{opt.condition}.pkz'), length)
test_data_extract   = seg_data(load_df(main_dir_colab + f'test_data_extract_{opt.condition}.pkz'), length)
test_data_c   = seg_data(load_df(main_dir_colab + f'test_c_{opt.condition}.pkz'), length)

test_label_1D  = seg_data(load_df(main_dir_colab + f'test_label_1D_{opt.condition}.pkz'), length)

