import numpy as np
import os
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from utils.extract_features import extracted_feature_of_signal
from os import path
from os.path import join

def convert_to_image(name_bearing, opt, type_data):
    '''
    This function is to get data and label from FPT points to later
    name_bearing: bearing name
    type_data: 1d, 2d, extract
    time: FPT
    type_: PHM, XJTU
    '''
    data = {'x': [], 'y': []}
    if type_data == '2d':
      print('-'*10, f'Convert to 2D data', '-'*10, '\n')
    else:
      print('-'*10, f'Maintain 1D data', '-'*10, '\n')
    
    num_files = len([i for i in os.listdir(name_bearing)])
    for i in range(num_files):
        name = f"{str(i+1)}.csv"
        file_ = join(name_bearing, name)
        if path.exists(file_):
            df = np.array(pd.read_csv(file_, header=None))[1:]
            coef_h = extract_feature_image(df, opt, type_data, feature_name='Horizontal_vibration_signals')
            coef_v = extract_feature_image(df, opt, type_data, feature_name='Vertical_vibration_signals')
            x_ = np.concatenate((coef_h, coef_v), axis=-1)
            if type_data=='1d' or type_data=='extract':
              x_ = x_.tolist()
            else:
              x_ = x_.tolist()
            if type_data == '1d':
              data['y'].append(compute_PCA(x_))
            data['x'].append(x_)
    
    data['x'] = np.array(data['x'])
    if type_data == '1d':
      data['y'] = np.array(data['y'])
      data['y'] = convert_1_to_0(data['y'])
        
    ############## 1D-data to extraction data #####################
    if type_data=='extract':
      print('-'*10, 'Convert to Extracted data', '-'*10, '\n')
      hor_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 0])
      ver_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 1])
      data_x = np.concatenate((hor_data, ver_data), axis=-1)
      data['x'] = data_x
    
    ############### scale data ##############################
    if type_data=='1d' or type_data=='extract':
        if opt.scaler == 'MinMaxScaler':
          scaler = MinMaxScaler
        if opt.scaler == 'MaxAbsScaler':
          scaler = MaxAbsScaler
        if opt.scaler == 'StandardScaler':
          scaler = StandardScaler
        if opt.scaler == 'RobustScaler':
          scaler = RobustScaler
        if opt.scaler == 'Normalizer':
          scaler = Normalizer
        if opt.scaler == 'QuantileTransformer':
          scaler = QuantileTransformer
        if opt.scaler == 'PowerTransformer':
          scaler = PowerTransformer

        if opt.scaler != None:
          hor_data = np.array(data['x'])[:, :, 0]
          ver_data = np.array(data['x'])[:, :, 1]
          print('-'*10, f'Use scaler: {opt.scaler}', '-'*10, '\n')
          if opt.scaler == 'FFT':
            hor_data = np.expand_dims(FFT(hor_data), axis=-1)
            ver_data = np.expand_dims(FFT(ver_data), axis=-1)
          elif opt.scaler == 'denoise':
            hor_data = denoise(hor_data)
            ver_data = denoise(ver_data)
          else:
            hor_data = scaler_transform(hor_data, scaler)
            ver_data = scaler_transform(ver_data, scaler)
          data_x = np.concatenate((hor_data, ver_data), axis=-1)
          data['x'] = data_x
        else:
          print('-'*10, 'Raw data', '-'*10, '\n')
          data['x'] = np.array(data['x'])
    if type_data == '1d':
      x_shape = data['x'].shape
      y_shape = data['y'].shape
      print(f'Train data shape: {x_shape}   Train label shape: {y_shape}\n')
    else:
      x_shape = data['x'].shape
      print(f'Train data shape: {x_shape}')
    return data


def extract_feature_image(df, opt, type_data, feature_name='horiz accel'):
    WAVELET_TYPE = 'morl'
    DATA_POINTS_PER_FILE=32768
    if feature_name == 'Horizontal_vibration_signals':
        data = df[:, 0].astype(np.float32)
    else:
        data = df[:, 1].astype(np.float32)

    WIN_SIZE = DATA_POINTS_PER_FILE//128
    
    if type_data == '2d':
        data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
        coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
        # transform to power and apply logarithm?!
        coef = np.log2(coef**2 + 0.001)
        coef = (coef - coef.min())/(coef.max() - coef.min())
        coef = np.expand_dims(coef, axis=-1)
    if type_data=='extract' or type_data=='1d':
        coef = np.expand_dims(data, axis=-1)
    return coef
