import os
import numpy as np
from train import parse_opt
from utils.tools import save_df, convert_to_image, getting_data
from utils.train_encoder import train_EC
from os.path import join, exists

opt = parse_opt()

main_dir_colab = join(opt.main_dir_colab, 'XJTU_data/XJTU-SY_Bearing_Datasets')
saved_dir = join(opt.main_dir_colab, 'XJTU_data/saved_data')

# Saving the converted data ==================================================================================
if os.path.exists(join(saved_dir, 'Bearing1_4_data_PCA_1d.npy')) == False:
  for type_data in opt.data_type:
    # Train data-------------------------------------------------------------------------
    Bearing1_1_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_1')
    Bearing1_1 = convert_to_image(Bearing1_1_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing1_1_data_PCA_' + type_data + '.npy'), Bearing1_1['x'])
    save_df(join(saved_dir, 'Bearing1_1_label_PCA.npy') , Bearing1_1['y'])

    Bearing1_2_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_2')
    Bearing1_2 = convert_to_image(Bearing1_2_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing1_2_data_PCA_' + type_data + '.npy'), Bearing1_2['x'])
    save_df(join(saved_dir, 'Bearing1_2_label_PCA.npy') , Bearing1_2['y'])

    Bearing1_3_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_3')
    Bearing1_3 = convert_to_image(Bearing1_3_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing1_3_data_PCA_' + type_data + '.npy'), Bearing1_3['x'])
    save_df(join(saved_dir, 'Bearing1_3_label_PCA.npy') , Bearing1_3['y'])

    Bearing1_5_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_5')
    Bearing1_5 = convert_to_image(Bearing1_5_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing1_5_data_PCA_' + type_data + '.npy'), Bearing1_5['x'])
    save_df(join(saved_dir, 'Bearing1_5_label_PCA.npy') , Bearing1_5['y'])

    Bearing2_1_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_1')
    Bearing2_1 = convert_to_image(Bearing2_1_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing2_1_data_PCA_' + type_data + '.npy'), Bearing2_1['x'])
    save_df(join(saved_dir, 'Bearing2_1_label_PCA.npy') , Bearing2_1['y'])

    Bearing2_2_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_2')
    Bearing2_2 = convert_to_image(Bearing2_2_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing2_2_data_PCA_' + type_data + '.npy'), Bearing2_2['x'])
    save_df(join(saved_dir, 'Bearing2_2_label_PCA.npy') , Bearing2_2['y'])

    Bearing2_3_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_3')
    Bearing2_3 = convert_to_image(Bearing2_3_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing2_3_data_PCA_' + type_data + '.npy'), Bearing2_3['x'])
    save_df(join(saved_dir, 'Bearing2_3_label_PCA.npy') , Bearing2_3['y'])

    Bearing2_4_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_4')
    Bearing2_4 = convert_to_image(Bearing2_4_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing2_4_data_PCA_' + type_data + '.npy'), Bearing2_4['x'])
    save_df(join(saved_dir, 'Bearing2_4_label_PCA.npy') , Bearing2_4['y'])

    Bearing2_5_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_5')
    Bearing2_5 = convert_to_image(Bearing2_5_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing2_5_data_PCA_' + type_data + '.npy'), Bearing2_5['x'])
    save_df(join(saved_dir, 'Bearing2_5_label_PCA.npy') , Bearing2_5['y'])

    Bearing3_1_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_1')
    Bearing3_1 = convert_to_image(Bearing3_1_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing3_1_data_PCA_' + type_data + '.npy'), Bearing3_1['x'])
    save_df(join(saved_dir, 'Bearing3_1_label_PCA.npy') , Bearing3_1['y'])

    Bearing3_3_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_3')
    Bearing3_3 = convert_to_image(Bearing3_3_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing3_3_data_PCA_' + type_data + '.npy'), Bearing3_3['x'])
    save_df(join(saved_dir, 'Bearing3_3_label_PCA.npy') , Bearing3_3['y'])

    Bearing3_4_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_4')
    Bearing3_4 = convert_to_image(Bearing3_4_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing3_4_data_PCA_' + type_data + '.npy'), Bearing3_4['x'])
    save_df(join(saved_dir, 'Bearing3_4_label_PCA.npy') , Bearing3_4['y'])

    Bearing3_5_path = join(main_dir_colab,  '40Hz10kN', 'Bearing3_5')
    Bearing3_5 = convert_to_image(Bearing3_5_path, opt, type_data, 'XJTU')
    save_df(join(saved_dir, 'Bearing3_5_data_PCA_' + type_data + '.npy'), Bearing3_5['x'])
    save_df(join(saved_dir, 'Bearing3_5_label_PCA.npy') , Bearing3_5['y'])

    print(f'\n Saving data in {opt.type} data set'+'-'*100)


test_1D, test_2D, test_extract, test_label_RUL, test_label_Con, test_idx = getting_data(saved_dir, opt.test_bearing, opt, get_index=True)
train_1D, train_2D, train_extract, train_label_RUL, train_label_Con = getting_data(saved_dir, opt.train_bearing, opt)

print('\n' + '#'*10 + f'Experimental case: {opt.case}'+ '#'*10 + '\n')

print(f'Shape of 1D training data: {train_1D.shape}')  
print(f'Shape of 1D test data: {test_1D.shape}\n')

print(f'Shape of 2D training data: {train_2D.shape}')  
print(f'Shape of 2D test data: {test_2D.shape}\n')

print(f'Shape of extract training data: {train_extract.shape}')  
print(f'Shape of extract test data: {test_extract.shape}\n')

print(f'Shape of training RUL label: {train_label_RUL.shape}')  
print(f'Shape of test RUL label: {test_label_RUL.shape}\n')

print(f'Shape of training Con label: {train_label_Con.shape}')  
print(f'Shape of test Con label: {test_label_Con.shape}\n')
