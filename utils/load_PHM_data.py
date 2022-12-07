import os
import numpy as np
from train import parse_opt
from utils.tools import  save_df, convert_to_image, getting_data

opt = parse_opt()
np.random.seed(1234)

# Link of original data ==================================================================================
train_dir =  '/PHM_data/Learning_set/'
test_dir = '/PHM_data/Test_set/'
saved_dir = '/PHM_data/saved_data/'

# FPT points of bearing sets ==================================================================================
FPT = {'Bearing1_1': 1314,
       'Bearing1_2': 826,
       'Bearing1_3': 1726,
       'Bearing1_4': 1082,
       'Bearing1_5': 2412,
       'Bearing1_6': 1631,
       'Bearing1_7': 2210}

# Saving the converted data ==================================================================================
if os.path.exists(saved_dir + 'Bearing1_1_' + '1d') == False:
  for type_data in opt.data_type:
    # Converting data-------------------------------------------------------------------------
    print('\n Saving data in XJTU data set'+'-'*100)
    Bearing1_1 = convert_to_image(train_dir + 'Bearing1_1', opt, type_data, FPT['Bearing1_1'], 'PHM')
    Bearing1_2 = convert_to_image(train_dir + 'Bearing1_2', opt, type_data, FPT['Bearing1_2'], 'PHM')
    Bearing1_3 = convert_to_image(test_dir  + 'Bearing1_3', opt, type_data, FPT['Bearing1_3'], 'PHM')
    Bearing1_4 = convert_to_image(test_dir  + 'Bearing1_4', opt, type_data, FPT['Bearing1_4'], 'PHM')
    Bearing1_5 = convert_to_image(test_dir  + 'Bearing1_5', opt, type_data, FPT['Bearing1_5'], 'PHM')
    Bearing1_6 = convert_to_image(test_dir  + 'Bearing1_6', opt, type_data, FPT['Bearing1_6'], 'PHM')
    Bearing1_7 = convert_to_image(test_dir  + 'Bearing1_7', opt, type_data, FPT['Bearing1_7'], 'PHM')
    
    # Save data and labels in different types------------------------------------------------
    save_df(saved_dir + 'Bearing1_1_data' + type_data, Bearing1_1['x'])
    save_df(saved_dir + 'Bearing1_2_data' + type_data, Bearing1_2['x'])
    save_df(saved_dir + 'Bearing1_3_data' + type_data, Bearing1_3['x'])
    save_df(saved_dir + 'Bearing1_4_data' + type_data, Bearing1_4['x'])
    save_df(saved_dir + 'Bearing1_5_data' + type_data, Bearing1_5['x'])
    save_df(saved_dir + 'Bearing1_6_data' + type_data, Bearing1_6['x'])
    save_df(saved_dir + 'Bearing1_7_data' + type_data, Bearing1_7['x'])

    save_df(saved_dir + 'Bearing1_1_label_RUL' , Bearing1_1['y'])
    save_df(saved_dir + 'Bearing1_2_label_RUL' , Bearing1_2['y'])
    save_df(saved_dir + 'Bearing1_3_label_RUL' , Bearing1_3['y'])
    save_df(saved_dir + 'Bearing1_4_label_RUL' , Bearing1_4['y'])
    save_df(saved_dir + 'Bearing1_5_label_RUL' , Bearing1_5['y'])
    save_df(saved_dir + 'Bearing1_6_label_RUL' , Bearing1_6['y'])
    save_df(saved_dir + 'Bearing1_7_label_RUL' , Bearing1_7['y'])


# Load saved bearing data ==================================================================================
test_1D, test_2D, test_extract, test_label_RUL = getting_data(saved_dir, opt.test_bearing, opt)
train_1D, train_2D, train_extract, train_label_RUL = getting_data(saved_dir, opt.train_bearing, opt)

print(f'Shape of 1D training data: {train_1D.shape}')  
print(f'Shape of 1D test data: {test_1D.shape}\n')

print(f'Shape of 2D training data: {train_2D.shape}')  
print(f'Shape of 2D test data: {test_2D.shape}\n')

print(f'Shape of extract training data: {train_extract.shape}')  
print(f'Shape of extract test data: {test_extract.shape}\n')

print(f'Shape of training RUL label: {train_label_RUL.shape}')  
print(f'Shape of test RUL label: {test_label_RUL.shape}\n')