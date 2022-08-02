import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from train import parse_opt
from utils.tools import load_df, save_df, extract_feature_image, convert_to_image, predict_time, seg_data, gen_rms

opt = parse_opt()


train_main_dir = '/content/drive/MyDrive/Khoa/XJTU_data/XJTU-SY_Bearing_Datasets/'

train_data_path_2D = train_main_dir + f'train_data_2D_{opt.condition}.pkz'
test_data_path_2D  = train_main_dir + f'test_data_2D_{opt.condition}.pkz'

train_data_path_1D  = train_main_dir + f'train_data_1D_{opt.condition}.pkz'
train_label_path_1D = train_main_dir + f'train_label_1D_{opt.condition}.pkz'
test_data_path_1D   = train_main_dir + f'test_data_1D_{opt.condition}.pkz'
test_label_path_1D  = train_main_dir + f'test_label_1D_{opt.condition}.pkz'

train_data_path_extract = train_main_dir + f'train_data_extract_{opt.condition}.pkz'
test_data_path_extract  = train_main_dir + f'test_data_extract_{opt.condition}.pkz'

train_c_path = train_main_dir + f'train_c_{opt.condition}.pkz'
test_c_path = train_main_dir + f'test_c_{opt.condition}.pkz'

if os.path.exists(test_data_path_2D) == False:
  for type_data in opt.data_type:
    # Train data-------------------------------------------------------------------------
    Bearing1_1_path = train_main_dir + '35Hz12kN/' + 'Bearing1_1'
    Bearing1_2_path = train_main_dir + '35Hz12kN/' + 'Bearing1_2'
    Bearing2_1_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_1'
    Bearing2_2_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_2'
    Bearing3_1_path = train_main_dir + '40Hz10kN/' + 'Bearing3_1'
    Bearing3_2_path = train_main_dir + '40Hz10kN/' + 'Bearing3_2'
    print('\n Training rul data'+'-'*100)
    Bearing1_1_data = convert_to_image(Bearing1_1_path, opt, type_data, 76, 'XJTU')
    Bearing1_2_data = convert_to_image(Bearing1_2_path, opt, type_data, 44, 'XJTU')
    Bearing2_1_data = convert_to_image(Bearing2_1_path, opt, type_data, 455, 'XJTU')
    Bearing2_2_data = convert_to_image(Bearing2_2_path, opt, type_data, 48, 'XJTU')
    Bearing3_1_data = convert_to_image(Bearing3_1_path, opt, type_data, 2344, 'XJTU')
    Bearing3_2_data = convert_to_image(Bearing3_2_path, opt, type_data, 1500, 'XJTU')
    
    if opt.condition in ['c_1', 'c_all']:
      train_data_rul  = train_data_1 = np.concatenate((Bearing1_1_data['x'], Bearing1_2_data['x']))
      train_label_rul = train_label_1 = np.concatenate((Bearing1_1_data['y'], Bearing1_2_data['y']))
      train_c_type    = train_c_type_1 = np.array([1.]*len(train_label_1))
    if opt.condition in ['c_2', 'c_all']:
      train_data_rul  = train_data_2 = np.concatenate((Bearing2_1_data['x'], Bearing2_2_data['x']))
      train_label_rul = train_label_2 = np.concatenate((Bearing2_1_data['y'], Bearing2_2_data['y']))
      train_c_type    = train_c_type_2 = np.array([2.]*len(train_label_2))
    if opt.condition in ['c_3', 'c_all']:
      train_data_rul  = train_data_3 = np.concatenate((Bearing3_1_data['x'], Bearing3_2_data['x']))
      train_label_rul = train_label_3 = np.concatenate((Bearing3_1_data['y'], Bearing3_2_data['y']))
      train_c_type    = train_c_type_3 = np.array([3.]*len(train_label_3))
    if opt.condition in ['c_all']:
      train_data_rul = np.concatenate((train_data_1, train_data_2, train_data_3))
      train_label_rul = np.concatenate((train_label_1, train_label_2, train_label_3))
      train_c_type = np.concatenate((train_c_type_1, train_c_type_2, train_c_type_3))

    # Test data---------------------------------------------------------------------------
    Bearing1_3_path = train_main_dir + '35Hz12kN/' + 'Bearing1_3'
    Bearing1_4_path = train_main_dir + '35Hz12kN/' + 'Bearing1_4'
    Bearing1_5_path = train_main_dir + '35Hz12kN/' + 'Bearing1_5'
    Bearing1_6_path = train_main_dir + '35Hz12kN/' + 'Bearing1_6'
    Bearing1_7_path = train_main_dir + '35Hz12kN/' + 'Bearing1_7'
    Bearing2_3_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_3'
    Bearing2_4_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_4'
    Bearing2_5_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_5'
    Bearing2_6_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_6'
    Bearing2_7_path = train_main_dir + '37.5Hz11kN/' + 'Bearing2_7'
    Bearing3_3_path = train_main_dir + '40Hz10kN/' + 'Bearing3_3'
    Bearing3_4_path = train_main_dir + '40Hz10kN/' + 'Bearing3_4'
    Bearing3_5_path = train_main_dir + '40Hz10kN/' + 'Bearing3_5'
    Bearing3_6_path = train_main_dir + '40Hz10kN/' + 'Bearing3_6'
    Bearing3_7_path = train_main_dir + '40Hz10kN/' + 'Bearing3_7'
    print('\n Test rul data'+'-'*100)
    Bearing1_3_data = convert_to_image(Bearing1_3_path, opt, type_data, 60, 'XJTU')
    Bearing1_4_data = convert_to_image(Bearing1_4_path, opt, type_data, 0, 'XJTU')
    Bearing1_5_data = convert_to_image(Bearing1_5_path, opt, type_data, 39, 'XJTU')

    Bearing2_3_data = convert_to_image(Bearing2_3_path, opt, type_data, 310, 'XJTU')
    Bearing2_4_data = convert_to_image(Bearing2_4_path, opt, type_data, 29, 'XJTU')
    Bearing2_5_data = convert_to_image(Bearing2_5_path, opt, type_data, 120, 'XJTU')

    Bearing3_3_data = convert_to_image(Bearing3_3_path, opt, type_data, 340, 'XJTU')
    Bearing3_4_data = convert_to_image(Bearing3_4_path, opt, type_data, 1418, 'XJTU')
    Bearing3_5_data = convert_to_image(Bearing3_5_path, opt, type_data, 9, 'XJTU')
    
    if opt.condition in ['c_1', 'c_all']:
      test_data_rul = test_data_1 = np.concatenate((Bearing1_3_data['x'], Bearing1_4_data['x'], Bearing1_5_data['x']))
      test_label_rul = test_label_1 = np.concatenate((Bearing1_3_data['y'], Bearing1_4_data['y'], Bearing1_5_data['y']))
      test_c_type    = test_c_type_1 = np.array([1.]*len(test_label_1))
    if opt.condition in ['c_2', 'c_all']:
      test_data_rul = test_data_2 = np.concatenate((Bearing2_3_data['x'], Bearing2_4_data['x'], Bearing2_5_data['x']))
      test_label_rul = test_label_2 = np.concatenate((Bearing2_3_data['y'], Bearing2_4_data['y'], Bearing2_5_data['y']))
      test_c_type    = test_c_type_2 = np.array([2.]*len(test_label_2))
    if opt.condition in ['c_3', 'c_all']:
      test_data_rul = test_data_3 = np.concatenate((Bearing3_3_data['x'], Bearing3_4_data['x'], Bearing3_5_data['x']))
      test_label_rul = test_label_3 = np.concatenate((Bearing3_3_data['y'], Bearing3_4_data['y'], Bearing3_5_data['y']))
      test_c_type    = test_c_type_3 = np.array([3.]*len(test_label_3))
    if opt.condition in ['c_all']:
      test_data_rul = np.concatenate((test_data_1, test_data_2, test_data_3))
      test_label_rul = np.concatenate((test_label_1, test_label_2, test_label_3))
      test_c_type = np.concatenate((test_c_type_1, test_c_type_2, test_c_type_3))
    
    # Save condition of data------------------------------------------------
    save_df(train_c_type, train_c_path)
    save_df(test_c_type, test_c_path)
    
    # Save data following to each type--------------------------------------
    if type_data == '1d':
      save_df(train_data_rul, train_data_path_1D)
      save_df(train_label_rul, train_label_path_1D)
      save_df(test_data_rul, test_data_path_1D)
      save_df(test_label_rul, test_label_path_1D)
    if type_data == 'extract':
      save_df(train_data_rul, train_data_path_extract)
      save_df(test_data_rul, test_data_path_extract)
    if type_data == '2d':
      save_df(test_data_rul, test_data_path_2D)
      save_df(train_data_rul, train_data_path_2D)
    print('#'*100)

train_data_rul_1D  = load_df(train_data_path_1D)
train_label_rul_1D = load_df(train_label_path_1D)
test_data_rul_1D   = load_df(test_data_path_1D)
test_label_rul_1D  = load_df(test_label_path_1D)

train_data_rul_2D  = load_df(train_data_path_2D)
test_data_rul_2D   = load_df(test_data_path_2D)

train_data_rul_extract  = load_df(train_data_path_extract)
test_data_rul_extract   = load_df(test_data_path_extract)

train_c   = load_df(train_c_path)
test_c  = load_df(test_c_path)

print(f'Train shape 1D: {train_data_rul_1D.shape}   {train_label_rul_1D.shape}')  
print(f'Test shape 1D: {test_data_rul_1D.shape}   {test_label_rul_1D.shape}\n')

print(f'Train shape 2D: {train_data_rul_2D.shape}')  
print(f'Test shape 2D: {test_data_rul_2D.shape} \n')

print(f'Train shape extract: {train_data_rul_extract.shape}')  
print(f'Test shape extract: {test_data_rul_extract.shape} \n')

print(f'shape of condition train and test: {train_c.shape}   {test_c.shape}\n')