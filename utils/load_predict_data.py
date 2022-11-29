import numpy as np
from train import parse_opt
from utils.tools import load_df, seg_data

opt = parse_opt()
np.random.seed(1234)

# FPT of each bearing in both data-sets



test_2D        = seg_data(load_df(opt.main_dir_colab + f'test_2D_{opt.condition}.pkz'), length)
test_1D        = seg_data(load_df(opt.main_dir_colab + f'test_1D_{opt.condition}.pkz'), length)
test_extract   = seg_data(load_df(opt.main_dir_colab + f'test_extract_{opt.condition}.pkz'), length)
test_label_Con = seg_data(load_df(opt.main_dir_colab + f'test_label_Con_{opt.condition}.pkz'), length)
test_label_RUL = seg_data(load_df(opt.main_dir_colab + f'test_label_RUL_{opt.condition}.pkz'), length)

