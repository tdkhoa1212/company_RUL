from train import parse_opt
from model.resnet import resnet_101
from model.LSTM import lstm_extracted_model, lstm_model
from model.MIX_1D_2D import mix_model_PHM, mix_model_XJTU
from utils.tools import all_matric_XJTU, all_matric_PHM, back_onehot
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter("ignore")
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True


opt = parse_opt()

# Loading data ######################################################
if opt.type == 'PHM':
  from utils.load_PHM_data import test_1D, test_2D, test_extract, test_label_RUL
else:
  from utils.load_XJTU_data import test_1D, test_2D, test_extract, test_label_Con, test_label_RUL

def Predict(data, opt):
  # Loading model ############################################################
  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')

  if opt.type == 'PHM':
    RUL = mix_model_PHM(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)
  else:
    Condition, RUL = mix_model_XJTU(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

  # Loading weights ############################################################
  weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')
  print(f'\nLoad weight: {weight_path}\n')
  network.load_weights(weight_path)
  
  # Prediction #####################################
  if opt.type == 'PHM':
    RUL = network.predict(data)
    return RUL 
  else:
    Condition, RUL = network.predict(data)
    return Condition, RUL 

def main(opt):
  result = {}
  name = opt.test_bearing[0]
  print(f'\nShape 1D of {name} data: {test_1D.shape}')
  print(f'Shape 2D of {name} data: {test_2D.shape}')
  if opt.type == 'PHM':
    RUL = Predict([test_1D, test_2D, test_extract])
  else:
    Condition, RUL = Predict([test_1D, test_2D, test_extract])

  if opt.type == 'PHM':
    r2, mae_, mse_ = all_matric_PHM(test_label_RUL, RUL, test_label_Con, Condition, opt)
  else:
    Condition = back_onehot(Condition)
    r2, mae_, mse_, acc = all_matric_XJTU(test_label_RUL, RUL, test_label_Con, Condition, opt)
    acc = round(acc*100, 4)

  mae_ = round(mae_, 4)
  rmse_ = round(mse_, 4)
  r2 = round(r2, 4)
  print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, RMSE: {rmse_}, Acc: {acc}-----')

  # Simulating the graphs --------------------------------------------------------
  plt.plot(test_label_RUL, c='b')
  plt.plot(RUL, c='r')
  plt.title(f'{name}: combination prediction.')
  plt.savefig(f'{name}.png')
  plt.close()

    
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  main()
