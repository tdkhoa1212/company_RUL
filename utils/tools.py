import numpy as np
import os
from keras import backend as K
import pandas as pd
import pickle as pkl
from numpy import save, load
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from utils.extract_features import extracted_feature_of_signal
from sklearn.metrics import r2_score, accuracy_score
from os import path
from os.path import join, exists


#---------------------- VALIDATION TOOLS------------------------------------------------
def to_onehot(label, m_po=3):
  new_label = np.zeros((len(label), m_po))
  for idx, i in enumerate(label):
    new_label[idx][int(i-1.)] = 1.
  return new_label

def back_onehot(label):
  a = []
  for i in label:
    a.append(np.argmax(i)+1)
  return np.array(a)

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_pred - K.mean(y_true) ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return SS_res/SS_tot 

def r2_numpy(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  np.sum(( y_pred - np.mean(y_true) )**2)
    SS_tot = np.sum(( y_true - np.mean(y_true) )**2)
    return SS_res/SS_tot 

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def rmse(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

def all_matric(y_true_rul, y_pred_rul):
    y_true_rul = np.squeeze(y_true_rul)
    y_pred_rul = np.squeeze(y_pred_rul)
    
    r2 = r2_score(y_true_rul, y_pred_rul)
    mae_ = mae(y_true_rul, y_pred_rul)
    rmse_ = rmse(y_true_rul, y_pred_rul)
    return r2, mae_, rmse_

    
#----------------------save_data.py------------------------------------------------
def load_file(path, save_path):
  a = [i for i in os.listdir(path)]
  all_file = []
  for i in range(len(a)):
    name = str(i)+'.csv'
    root = join(path, name)
    if exists(root):
      df=pd.read_csv(root, header=None, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])
      df = np.expand_dims(np.array(df)[1:], axis=0).astype(np.float32)
      if all_file == []:
        all_file = df
      else:
        all_file = np.concatenate((all_file, df))
  save_df(all_file, save_path)

def read_data_as_df(base_dir):
  '''
  saves each file in the base_dir as a df and concatenate all dfs into one
  '''
  if base_dir[-1]!='/':
    base_dir += '/'

  dfs=[]
  for f in sorted(os.listdir(base_dir)):
    if f[:3] == 'acc':
      df=pd.read_csv(base_dir+f, header=None, names=['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel'])
      dfs.append(df)
  return pd.concat(dfs)

def process(base_dir, out_file):
  '''
  dumps combined dataframes into pkz (pickle) files for faster retreival
  '''
  df=read_data_as_df(base_dir)
  # assert df.shape[0]==len(os.listdir(base_dir))*DATA_POINTS_PER_FILE
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
  print('{0} saved'.format(out_file))
  print(f'Shape: {df.shape}\n')

#----------------------Load_data.py------------------------------------------------
def load_df(pkz_file):
    df = load(pkz_file)
    return df

def save_df(out_file, df):
  save(out_file, df)

def scaler(signal, scale_method):
  scale = scale_method().fit(signal)
  return scale.transform(signal), scale

def scaler_transform(signals, scale_method):
  data = []
  scale = scale_method()
  for signal in signals:
    if len(signal.shape) < 2:
      signal = np.expand_dims(signal, axis=-1)
    data.append(scale.fit_transform(signal))
  return np.array(data)

# def extract_feature_image(df, opt, type_data, feature_name='horiz accel'):
#     WAVELET_TYPE = 'morl'
#     DATA_POINTS_PER_FILE=32768
#     if feature_name == 'Horizontal_vibration_signals':
#         data = df[:, 0].astype(np.float32)
#     else:
#         data = df[:, 1].astype(np.float32)

#     WIN_SIZE = DATA_POINTS_PER_FILE//128
    
#     if type_data == '2d':
#         data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
#         coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
#         # transform to power and apply logarithm?!
#         coef = np.log2(coef**2 + 0.001)
#         coef = (coef - coef.min())/(coef.max() - coef.min())
#         coef = np.expand_dims(coef, axis=-1)
#     if type_data=='extract' or type_data=='1d':
#         coef = np.expand_dims(data, axis=-1)
#     return coef

def denoise(signals):
    all_signal = []
    for x in signals:
        L1, L2, L3 = pywt.wavedec(x, 'coif7', level=2)
        all_ = np.expand_dims(np.concatenate((L1, L2, L3)), axis=0)
        if all_signal == []:
          all_signal = all_
        else:
          all_signal = np.concatenate((all_signal, all_))
        # all_signal.append(nr.reduce_noise(y=x, sr=2559, hop_length=20, time_constant_s=0.1, prop_decrease=0.5, freq_mask_smooth_hz=25600))
    return np.expand_dims(all_signal, axis=-1)

def compute_PCA(x_all):
  label = []
  for x in x_all:
    pca = PCA(n_components=1)
    pca.fit(x)
    label.append(pca.singular_values_[0])
  return np.array(label)

def convert_1_to_0(data):
    if np.min(data) != np.max(data):
      f_data = (data - np.min(data))/(np.max(data) - np.min(data))
    else:
      f_data = np.ones_like(data)
    return f_data

# def convert_to_image(name_bearing, opt, type_data):
#     '''
#     This function is to get data and label from FPT points to later
#     name_bearing: bearing name
#     type_data: 1d, 2d, extract
#     time: FPT
#     type_: PHM, XJTU
#     '''
#     data = {'x': [], 'y': []}
#     if type_data == '2d':
#       print('-'*10, f'Convert to 2D data', '-'*10, '\n')
#     else:
#       print('-'*10, f'Maintain 1D data', '-'*10, '\n')
    
#     num_files = len([i for i in os.listdir(name_bearing)])
#     for i in range(num_files):
#         name = f"{str(i+1)}.csv"
#         file_ = join(name_bearing, name)
#         if path.exists(file_):
#             df = np.array(pd.read_csv(file_, header=None))[1:]
#             coef_h = extract_feature_image(df, opt, type_data, feature_name='Horizontal_vibration_signals')
#             coef_v = extract_feature_image(df, opt, type_data, feature_name='Vertical_vibration_signals')
#             x_ = np.concatenate((coef_h, coef_v), axis=-1)
#             if type_data=='1d' or type_data=='extract':
#               x_ = x_.tolist()
#             else:
#               x_ = x_.tolist()
#             if type_data == '1d':
#               data['y'].append(compute_PCA(x_))
#             data['x'].append(x_)
    
#     data['x'] = np.array(data['x'])
#     if type_data == '1d':
#       data['y'] = np.array(data['y'])
#       data['y'] = convert_1_to_0(data['y'])
        
#     ############## 1D-data to extraction data #####################
#     if type_data=='extract':
#       print('-'*10, 'Convert to Extracted data', '-'*10, '\n')
#       hor_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 0])
#       ver_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 1])
#       data_x = np.concatenate((hor_data, ver_data), axis=-1)
#       data['x'] = data_x
    
#     ############### scale data ##############################
#     if type_data=='1d' or type_data=='extract':
#         if opt.scaler == 'MinMaxScaler':
#           scaler = MinMaxScaler
#         if opt.scaler == 'MaxAbsScaler':
#           scaler = MaxAbsScaler
#         if opt.scaler == 'StandardScaler':
#           scaler = StandardScaler
#         if opt.scaler == 'RobustScaler':
#           scaler = RobustScaler
#         if opt.scaler == 'Normalizer':
#           scaler = Normalizer
#         if opt.scaler == 'QuantileTransformer':
#           scaler = QuantileTransformer
#         if opt.scaler == 'PowerTransformer':
#           scaler = PowerTransformer

#         if opt.scaler != None:
#           hor_data = np.array(data['x'])[:, :, 0]
#           ver_data = np.array(data['x'])[:, :, 1]
#           print('-'*10, f'Use scaler: {opt.scaler}', '-'*10, '\n')
#           if opt.scaler == 'FFT':
#             hor_data = np.expand_dims(FFT(hor_data), axis=-1)
#             ver_data = np.expand_dims(FFT(ver_data), axis=-1)
#           elif opt.scaler == 'denoise':
#             hor_data = denoise(hor_data)
#             ver_data = denoise(ver_data)
#           else:
#             hor_data = scaler_transform(hor_data, scaler)
#             ver_data = scaler_transform(ver_data, scaler)
#           data_x = np.concatenate((hor_data, ver_data), axis=-1)
#           data['x'] = data_x
#         else:
#           print('-'*10, 'Raw data', '-'*10, '\n')
#           data['x'] = np.array(data['x'])
#     if type_data == '1d':
#       x_shape = data['x'].shape
#       y_shape = data['y'].shape
#       print(f'Train data shape: {x_shape}   Train label shape: {y_shape}\n')
#     else:
#       x_shape = data['x'].shape
#       print(f'Train data shape: {x_shape}')
#     return data

def FFT(signals):
  fft_data = []
  for signal in signals:
    signal = np.fft.fft(signal)
    signal = np.abs(signal) / len(signal)
    signal = signal[range(signal.shape[0] // 2)]
    signal = np.expand_dims(signal, axis=0)
    if fft_data == []:
      fft_data = signal
    else:
      fft_data = np.concatenate((fft_data, signal))
  return fft_data

# ---------------------- Load_predict_data.py----------------------
def seg_data(data, length):
  all_data = {}
  num=0
  for name in length:
    all_data[name] = data[num: num+length[name]]
    num += length[name]
  return all_data

# ----------------------Creating label----------------------
def gen_rms(col):
    return np.squeeze(np.sqrt(np.mean(col**2)))

def getting_data(saved_dir, bearing_list, opt, get_index=False):
  _1D = []
  _2D = []
  extract = []

  # Creating empty folder to catch labels--------------
  label_RUL_all = []

  if get_index:
    idx = {}

  # Arranging data and labels in scheme---------------
  for name in bearing_list:
    for type_data in opt.data_type:
      # Loading data and labels-----------------------
      data     = load_df(join(saved_dir, name + '_data_PCA_'  + type_data + '.npy'))
      label_RUL= load_df(join(saved_dir, name + '_label_PCA.npy'))
      if get_index:
        idx[name] = label_RUL.shape[0]

      # Getting 1D data and labels-----------------------------------
      if type_data == '1d':
        if _1D == []:
          _1D = data 
          label_RUL_all = label_RUL
        else:
          _1D = np.concatenate((_1D, data))
          label_RUL_all = np.concatenate((label_RUL_all, label_RUL))

      # Getting 2D data --------------------------------------------
      elif type_data == '2d':
        if _2D == []:
          _2D = data 
        else:
          _2D = np.concatenate((_2D, data))
      
      # Getting extract data ---------------------------------------
      else:
        if extract == []:
          extract = data 
        else:
          extract = np.concatenate((extract, data))

  if get_index:
    return _1D, _2D, extract, label_RUL_all, idx
  else:
    return _1D, _2D, extract, label_RUL_all

def load_RUL_data(name, opt):
  short_data = pd.read_csv(join(opt.main_dir_colab, f'{name}/short.csv'))
  ##################### Stick time's colum and label's column #####################
  datetime = np.array(short_data['datetime'])
  label = np.array(short_data['label'])
  time_label = {}

  for i, each_datetime in enumerate(datetime):
    name_0 = each_datetime.split('/')
    h = name_0[2][5:7]
    name = f'{name_0[0]}_{name_0[1]}_{name_0[2][:4]}__{h}h'
    time_label[name] = label[i]

  ################# data and label creation ##############################
  path = join(opt.main_dir_colab, f'{name}/long')
  data = []
  label = []

  hours = list(range(0, 24))
  days = list(range(1, 32))
  months = list(range(1, 13))
  years = [2021, 2022]
  name_files = {}
  for i in os.listdir(path):
    name_files[i[:15]] = i

  name_all = list(name_files.keys())

  for y in years:
    for m in months:
      for d in days:
        for h in hours:
          name = f'{str(d).zfill(2)}_{str(m).zfill(2)}_{str(y).zfill(2)}__{str(h).zfill(2)}h'
          if name in name_all:
            e_file = os.path.join(path, name_files[name])
            a = pd.read_csv(e_file)
            main_data = np.array(a)[:, 1:]
            if name in list(time_label):
              data.append(main_data.tolist())
              label.append(time_label[name])
  return np.array(data), np.array(label)

def extract_feature_image(data_2c, type_data, feature_name):
    WAVELET_TYPE = 'morl'
    DATA_POINTS_PER_FILE=32768
    if feature_name == 'Horizontal_vibration_signals':
        data = data_2c[:, 0].astype(np.float32)
    else:
        data = data_2c[:, 1].astype(np.float32)

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

def convert_to_image(name_bearing, opt, type_data):
    '''
    '''
    data = {'x': [], 'y': []}
    #------------------------------------------------------ 2D -------------------------------------------
    if type_data == '2d':
      print('-'*10, f'Convert to 2D data', '-'*10, '\n')
    else:
      print('-'*10, f'Maintain 1D data', '-'*10, '\n')
    
    data_all, _ = load_RUL_data(name_bearing, opt)
    for data_2c in range(data_all):
        coef_h = extract_feature_image(data_2c, type_data, 'Horizontal_vibration_signals')
        coef_v = extract_feature_image(data_2c, type_data, 'Vertical_vibration_signals')
        x_ = np.concatenate((coef_h, coef_v), axis=-1)
        if type_data=='1d' or type_data=='extract':
          x_ = x_.tolist()
        else:
          x_ = x_.tolist()
        if type_data == '1d':
          data['y'].append(compute_PCA(x_))
        data['x'].append(x_)

    if type_data == '1d':
      data['y'] = convert_1_to_0(np.array(data['y']))
    data['x'] = np.array(data['x'])
    
    #------------------------------------------------------ 1D -------------------------------------------
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