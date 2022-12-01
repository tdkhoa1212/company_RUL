import os
import numpy as np
from train import parse_opt
from utils.tools import load_df, save_df, convert_to_image

opt = parse_opt()
np.random.seed(1234)

main_dir_colab = opt.main_dir_colab
train_dir =  '/PHM_data/Learning_set/'
test_dir = '/PHM_data/Test_set/'
saved_dir = '/PHM_data/saved_data/'

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
    print('\nSaving all forms of data'+'-'*100)
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

    save_df(saved_dir + 'Bearing1_1_label' + type_data, Bearing1_1['y'])
    save_df(saved_dir + 'Bearing1_2_label' + type_data, Bearing1_2['y'])
    save_df(saved_dir + 'Bearing1_3_label' + type_data, Bearing1_3['y'])
    save_df(saved_dir + 'Bearing1_4_label' + type_data, Bearing1_4['y'])
    save_df(saved_dir + 'Bearing1_5_label' + type_data, Bearing1_5['y'])
    save_df(saved_dir + 'Bearing1_6_label' + type_data, Bearing1_6['y'])
    save_df(saved_dir + 'Bearing1_7_label' + type_data, Bearing1_7['y'])

# Loading the converted data ==================================================================================
Bearing1_1_data = convert_to_image(train_dir + 'Bearing1_1', opt, type_data, FPT['Bearing1_1'], 'PHM')
Bearing1_2_data = convert_to_image(train_dir + 'Bearing1_2', opt, type_data, FPT['Bearing1_2'], 'PHM')
Bearing1_3_data = convert_to_image(test_dir  + 'Bearing1_3', opt, type_data, FPT['Bearing1_3'], 'PHM')
Bearing1_4_data = convert_to_image(test_dir  + 'Bearing1_4', opt, type_data, FPT['Bearing1_4'], 'PHM')
Bearing1_5_data = convert_to_image(test_dir  + 'Bearing1_5', opt, type_data, FPT['Bearing1_5'], 'PHM')
Bearing1_6_data = convert_to_image(test_dir  + 'Bearing1_6', opt, type_data, FPT['Bearing1_6'], 'PHM')
Bearing1_7_data = convert_to_image(test_dir  + 'Bearing1_7', opt, type_data, FPT['Bearing1_7'], 'PHM')


def getting_data(bearing_list):
  train_1D = []
  test_1D = []

  train_2D = []
  test_2D = []

  train_extract = []
  test_extract = []

  test_label_RUL = []
  
  for name in bearing_list:
    for type_data in opt.data_type:
      if type_data == '1d':
        Bearing_data  = load_df(saved_dir + name + '_data'  + type_data)
        Bearing_label = load_df(saved_dir + name + '_label' + type_data)

        if train_1D == []:
          train_1D = Bearing_data 
          label = Bearing_label 
        else:
          data = np.concatenate((data, Bearing_data))
          label = np.concatenate((label, Bearing_label))

  Bearing1_2_data = load_df(saved_dir  + 'Bearing1_2_data'  + type_data)
  Bearing1_2_label = load_df(saved_dir + 'Bearing1_2_label' + type_data)

  Bearing1_3_data = load_df(saved_dir  + 'Bearing1_3_data'  + type_data)
  Bearing1_3_label = load_df(saved_dir + 'Bearing1_3_label' + type_data)

  Bearing1_4_data = load_df(saved_dir  + 'Bearing1_4_data'  + type_data)
  Bearing1_4_label = load_df(saved_dir + 'Bearing1_4_label' + type_data)

  Bearing1_5_data = load_df(saved_dir  + 'Bearing1_5_data'  + type_data)
  Bearing1_5_label = load_df(saved_dir + 'Bearing1_5_label' + type_data)

  Bearing1_6_data = load_df(saved_dir  + 'Bearing1_6_data'  + type_data)
  Bearing1_6_label = load_df(saved_dir + 'Bearing1_6_label' + type_data)

  Bearing1_7_data = load_df(saved_dir  + 'Bearing1_7_data'  + type_data)
  Bearing1_7_label = load_df(saved_dir + 'Bearing1_7_label' + type_data)

  test_label_RUL = np.concatenate((Bearing1_1_label, Bearing1_2_label, Bearing1_3_label, Bearing1_4_label, Bearing1_5_label, Bearing1_6_label, Bearing1_7_label))
  data = np.concatenate((Bearing1_1_data, Bearing1_2_data, Bearing1_3_data, Bearing1_4_data, Bearing1_5_data, Bearing1_6_data, Bearing1_7_data))
  if type_data == '1d':
    train_1D = data
  if type_data == '2d':
     

print(f'Train shape 1D: {train_data_rul_1D.shape}   {train_label_rul_1D.shape}')  
print(f'Test shape 1D: {test_data_rul_1D.shape}   {test_label_rul_1D.shape}\n')

print(f'Train shape 2D: {train_data_rul_2D.shape}')  
print(f'Test shape 2D: {test_data_rul_2D.shape} \n')

print(f'Train shape extract: {train_data_rul_extract.shape}')  
print(f'Test shape extract: {test_data_rul_extract.shape} \n')

print(f'shape of condition train and test: {train_c.shape}   {test_c.shape}\n')
