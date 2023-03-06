import os
from train import parse_opt
from utils.tools import save_df, convert_to_image, getting_data
from os.path import join

opt = parse_opt()

saved_dir = join(opt.main_dir_colab, 'saved_data')

# Saving the converted data ==================================================================================
if os.path.exists(os.getcwd() + join(saved_dir, 'Data001_data_2d.npy')) == False:
  for type_data in opt.data_type:
    # Train data-------------------------------------------------------------------------
    Data001_path = join(opt.main_dir_colab, 'data001')
    Data001 = convert_to_image(Data001_path, opt, type_data)
    save_df(os.getcwd() + join(saved_dir, 'Data001_data_' + type_data + '.npy'), Data001['x'])

    Data002_path = join(opt.main_dir_colab, 'data002')
    Data002 = convert_to_image(Data002_path, opt, type_data)
    save_df(os.getcwd() + join(saved_dir, 'Data002_data_' + type_data + '.npy'), Data002['x'])

    Data003_path = join(opt.main_dir_colab, 'data003')
    Data003 = convert_to_image(Data003_path, opt, type_data)
    save_df(os.getcwd() + join(saved_dir, 'Data003_data_' + type_data + '.npy'), Data003['x'])

    Data004_path = join(opt.main_dir_colab, 'data004')
    Data004 = convert_to_image(Data004_path, opt, type_data)
    save_df(os.getcwd() + join(saved_dir, 'Data004_data_' + type_data + '.npy'), Data004['x'])

    Data005_path = join(opt.main_dir_colab, 'data005')
    Data005 = convert_to_image(Data005_path, opt, type_data)
    save_df(os.getcwd() + join(saved_dir, 'Data005_data_' + type_data + '.npy'), Data005['x'])

    if type_data == '1d':
      save_df(os.getcwd() + join(saved_dir, 'Data001_label.npy') , Data001['y'])
      save_df(os.getcwd() + join(saved_dir, 'Data002_label.npy') , Data002['y'])
      save_df(os.getcwd() + join(saved_dir, 'Data003_label.npy') , Data003['y'])
      save_df(os.getcwd() + join(saved_dir, 'Data004_label.npy') , Data004['y'])
      save_df(os.getcwd() + join(saved_dir, 'Data005_label.npy') , Data005['y'])

    print(f'\n Saving data in {type_data} data set'+'-'*100)


test_1D, test_2D, test_extract, test_label_RUL, test_idx = getting_data(saved_dir, opt.test_bearing, opt, get_index=True)
train_1D, train_2D, train_extract, train_label_RUL = getting_data(saved_dir, opt.train_bearing, opt)


print(f'Shape of 1D training data: {train_1D.shape}')  
print(f'Shape of 1D test data: {test_1D.shape}\n')

print(f'Shape of 2D training data: {train_2D.shape}')  
print(f'Shape of 2D test data: {test_2D.shape}\n')

print(f'Shape of extract training data: {train_extract.shape}')  
print(f'Shape of extract test data: {test_extract.shape}\n')

print(f'Shape of training RUL label: {train_label_RUL.shape}')  
print(f'Shape of test RUL label: {test_label_RUL.shape}\n')

