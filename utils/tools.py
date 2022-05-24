import numpy as np
from keras import backend as K

#----------------------#### General ####------------------------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
def accuracy_m(y_true, y_pred):
  correct = 0
  total = 0
  for i in range(len(y_true)):
      act_label = np.argmax(y_true[i]) # act_label = 1 (index)
      pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
      if(act_label == pred_label):
          correct += 1
      total += 1
  accuracy = (correct/total)
  return accuracy

def to_onehot(label):
  new_label = np.zeros((len(label), np.max(label)+1))
  for idx, i in enumerate(label):
    new_label[idx][i] = 1.
  return new_label

#----------------------save_data.py------------------------------------------------
def read_data_as_df(base_dir):
  '''
  saves each file in the base_dir as a df and concatenate all dfs into one
  '''
  if base_dir[-1]!='/':
    base_dir += '/'

  dfs=[]
  for f in sorted(os.listdir(base_dir)):
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
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

def df_row_ind_to_data_range(ind):
    return (DATA_POINTS_PER_FILE*ind, DATA_POINTS_PER_FILE*(ind+1))

def extract_feature_image(ind, feature_name='horiz accel'):
    data_range = df_row_ind_to_data_range(ind)
    data = df[feature_name].values[data_range[0]: data_range[1]]
    # use window to process(= prepare, develop) 1D signal
    data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
    # perform CWT on 1D data(= 1D array)
    coef, _ = pywt.cwt(data, np.linspace(1,128,128), WAVELET_TYPE)
    # transform to power and apply logarithm?!
    coef = np.log2(coef**2 + 0.001)
    # normalize coef
    coef = (coef - coef.min())/(coef.max() - coef.min()) 
    return coef

def convert_to_image(pkz_dir):
    df = load_df(pkz_dir)
    no_of_rows = df.shape[0]
    no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
    print(f'pkz file length: {no_of_rows}, total subsequence data: {no_of_files}')
    
    data = {'x': [], 'y': []}
    for i in range(0, no_of_files):
        coef_h = extract_feature_image(i, feature_name='horiz accel')
        coef_v = extract_feature_image(i, feature_name='vert accel')
        x_ = np.array([coef_h, coef_v])
        y_ = i/(no_of_files-1)
        data['x'].append(x_)
        data['y'].append(y_)
    data['x']=np.array(data['x'])
    data['y']=np.array(data['y'])

    assert data['x'].shape==(no_of_files, 2, 128, 128)
    x_shape = data['x'].shape
    y_shape = data['y'].shape
    print(f'Train data shape: {x_shape}   Train label shape: {y_shape}\n')
    return data
