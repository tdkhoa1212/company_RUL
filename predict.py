from train import parse_opt
from model.resnet import resnet_101
from model.MIX_1D_2D import mix_model
from utils.tools import get_pre_data, load_df
from model.LSTM import lstm_extracted_model, lstm_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter("ignore")
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True


opt = parse_opt()

def Predict(data, opt):
  # Loading model ############################################################
  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')

  RUL = mix_model(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
  network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=RUL)

  # Loading weights ############################################################
  weight_path = 'results/model_PCA'
  network.load_weights(weight_path)
  
  # Prediction #####################################
  RUL = network.predict(data)
  return RUL 


def main(opt):
  path = "/test_data"
  for name in os.listdir(path):
    file = os.path.join(path, name)
    t_1D = get_pre_data(file, opt, "1d")
    t_2D = get_pre_data(file, opt, "2d")
    t_extract = get_pre_data(file, opt, "extract")
    RUL = Predict([t_1D, t_2D, t_extract], opt)[0][0]
    print(f"\nRUL of the {name}: {RUL}")

def folder_main(opt):
  t_1D = load_df("data/saved_data/Data001_data_1d.npy")
  t_2D = load_df("data/saved_data/Data001_data_2d.npy")
  t_extract = load_df("data/saved_data/Data001_data_extract.npy")
  RUL = Predict([t_1D, t_2D, t_extract], opt)
  labels = load_df("data/saved_data/Data001_label.npy")
  plt.plot(RUL, c='r', label='pre_RUL')
  plt.plot(labels, c='b', label='label')
  plt.savefig('results/image.png')
  plt.show()
    
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  # main(opt)
  folder_main(opt)
