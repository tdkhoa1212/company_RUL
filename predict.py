from train import parse_opt
from model.resnet import resnet_101
from model.MIX_1D_2D import mix_model

from model.LSTM import lstm_extracted_model, lstm_model
from utils.tools import all_matric
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from os.path import join
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter("ignore")
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True


opt = parse_opt()

# Loading data ######################################################
from utils.load_data import test_1D, test_2D, test_extract, test_label_RUL, test_idx

def Predict(data, opt):
  # Loading model ############################################################
  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')

  RUL = mix_model(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
  network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)

  # Loading weights ############################################################
  weight_path = os.path.join(opt.save_dir, f'model_PCA')
  print(f'\nLoad weight: {weight_path}\n')
  network.load_weights(weight_path)
  
  # Prediction #####################################
  RUL = network.predict(data)
  return RUL 


def main(opt):
  num = 0
  for name in opt.test_bearing:
    t_1D, t_2D, t_extract = test_1D[num: num+test_idx[name]], test_2D[num: num+test_idx[name]], test_extract[num: num+test_idx[name]]
    print(f'\nShape 1D of {name} data: {t_1D.shape}')
    print(f'Shape 2D of {name} data: {t_2D.shape}')

    RUL = Predict([t_1D, t_2D, t_extract], opt)


    t_label_RUL = test_label_RUL[num: num+test_idx[name]]
    num += test_idx[name]
    r2, mae_, mse_ = all_matric(t_label_RUL, RUL)
   
    mae_ = round(mae_, 4)
    rmse_ = round(mse_, 4)
    r2 = round(r2, 4)

    print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, RMSE: {rmse_}-----')

    # Simulating the graphs --------------------------------------------------------
    plt.plot(t_label_RUL, c='b', label="True")
    plt.plot(RUL, c='r', label="Prediction")
    plt.legend()
    plt.title(f'{name}')
    plt.savefig(join(opt.save_dir, 'XJTU_PCA', f'{name}.png'))
    plt.close()

    
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  main(opt)
