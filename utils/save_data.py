import os
from utils.tools import process

def start_save_data(opt):
  train_names = ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']
  test_names = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7', 'Bearing3_3']

  main_dir_colab = opt.main_dir_colab
  train_main_dir = main_dir_colab + 'Learning_set/'
  test_main_dir = main_dir_colab + 'Test_set/'
  if os.path.exists(train_main_dir + train_names[0] + '.pkz')==False:
    for train_name in train_names:
      base_dir = train_main_dir + train_name + '/'
      out_file = train_main_dir + train_name + '.pkz'
      process(base_dir, out_file)
    for test_name in test_names:
      base_dir = test_main_dir + test_name + '/'
      out_file = test_main_dir + test_name + '.pkz'
      process(base_dir, out_file)
